// Memory / GPIO benchmark. Prints machine-readable lines:
//   BENCH,<memory>,<test>,<size_bytes>,<value>,<unit>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "driver/gpio.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_flash.h"
#include "esp_timer.h"
#include "esp_cpu.h"
#include "esp_random.h"
#include "soc/gpio_struct.h"
#include "esp_clk_tree.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define MIN_US 300000          // each measurement runs at least this long
#define TEST_GPIO 4

static volatile uint32_t sink;

static void emit(const char *mem, const char *test, size_t size, double v, const char *unit)
{
    printf("BENCH,%s,%s,%u,%.2f,%s\n", mem, test, (unsigned)size, v, unit);
    vTaskDelay(1);
}

// Run fn(arg) repeatedly until MIN_US elapsed; return MB/s for `bytes` per call.
typedef void (*pass_fn)(void *buf, size_t n, void *aux);
static double mbps(pass_fn fn, void *buf, size_t n, void *aux, size_t bytes_per_pass)
{
    int64_t t0 = esp_timer_get_time(), now;
    uint64_t passes = 0;
    do { fn(buf, n, aux); passes++; now = esp_timer_get_time(); } while (now - t0 < MIN_US);
    double secs = (now - t0) / 1e6;
    return (double)passes * bytes_per_pass / secs / (1024.0 * 1024.0);
}

static void rd32(void *b, size_t n, void *aux)
{
    const volatile uint32_t *p = b; uint32_t s = 0;
    for (size_t i = 0; i < n / 4; i += 8)
        s += p[i] + p[i+1] + p[i+2] + p[i+3] + p[i+4] + p[i+5] + p[i+6] + p[i+7];
    sink = s;
}
static void wr32(void *b, size_t n, void *aux)
{
    volatile uint32_t *p = b;
    for (size_t i = 0; i < n / 4; i += 8) {
        p[i] = i; p[i+1] = i; p[i+2] = i; p[i+3] = i; p[i+4] = i; p[i+5] = i; p[i+6] = i; p[i+7] = i;
    }
}
static void cpy(void *b, size_t n, void *aux) { memcpy((uint8_t *)b + n / 2, b, n / 2); }

// Pointer chase over a random single cycle (Sattolo), one node per 64 bytes:
// measures latency. Built in place so no extra buffer is needed.
static double latency_ns(uint32_t *buf, size_t bytes)
{
    size_t nodes = bytes / 64;
    const size_t stride = 16;                    // 16 words = 64 bytes
    for (size_t i = 0; i < nodes; i++) buf[i * stride] = i;
    for (size_t i = nodes - 1; i > 0; i--) {
        size_t j = esp_random() % i;             // j < i gives one cycle
        uint32_t t = buf[i * stride]; buf[i * stride] = buf[j * stride]; buf[j * stride] = t;
    }
    for (size_t i = 0; i < nodes; i++) buf[i * stride] *= stride;
    volatile uint32_t *p = buf; uint32_t idx = 0;
    const int steps = 200000;
    int64_t t0 = esp_timer_get_time();
    for (int i = 0; i < steps; i++) idx = p[idx];
    int64_t dt = esp_timer_get_time() - t0;
    sink = idx;
    return dt * 1000.0 / steps;
}

static void sweep(const char *mem, void *buf, size_t max, bool writable)
{
    for (size_t sz = 4096; sz <= max; sz *= 2) {
        emit(mem, "read32_sweep", sz, mbps(rd32, buf, sz, NULL, sz), "MB/s");
        if (writable) emit(mem, "write32_sweep", sz, mbps(wr32, buf, sz, NULL, sz), "MB/s");
    }
}

static void bench_ram(const char *mem, uint32_t caps, size_t sweep_max, size_t big)
{
    uint32_t *buf = heap_caps_malloc(big, caps);
    if (!buf) { printf("BENCH,%s,alloc_failed,%u,0,\n", mem, (unsigned)big); return; }
    memset(buf, 0x5a, big);
    sweep(mem, buf, sweep_max, true);
    emit(mem, "read32", big, mbps(rd32, buf, big, NULL, big), "MB/s");
    emit(mem, "write32", big, mbps(wr32, buf, big, NULL, big), "MB/s");
    emit(mem, "memcpy", big, mbps(cpy, buf, big, NULL, big / 2), "MB/s");
    emit(mem, "latency", big, latency_ns(buf, big), "ns");
    free(buf);
}

static void bench_flash(void)
{
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_APP, ESP_PARTITION_SUBTYPE_ANY, NULL);
    if (!part) { printf("BENCH,flash,no_partition,0,0,\n"); return; }
    size_t big = 1024 * 1024;
    const void *map; esp_partition_mmap_handle_t h;
    if (esp_partition_mmap(part, 0, big, ESP_PARTITION_MMAP_DATA, &map, &h) == ESP_OK) {
        sweep("flash_mmap", (void *)map, big, false);
        emit("flash_mmap", "read32", big, mbps(rd32, (void *)map, big, NULL, big), "MB/s");
        esp_partition_munmap(h);
    }
    // Direct (uncached) reads into internal RAM
    size_t chunk = 64 * 1024;
    uint8_t *dst = heap_caps_malloc(chunk, MALLOC_CAP_INTERNAL);
    if (dst) {
        int64_t t0 = esp_timer_get_time(), now; uint64_t n = 0;
        do { esp_flash_read(NULL, dst, part->address, chunk); n++; now = esp_timer_get_time(); } while (now - t0 < MIN_US);
        emit("flash_direct", "read", chunk, n * chunk / ((now - t0) / 1e6) / (1024.0 * 1024.0), "MB/s");
        free(dst);
    }
}

static void bench_gpio(void)
{
    gpio_set_direction(TEST_GPIO, GPIO_MODE_OUTPUT);
    const int n = 200000;
    int64_t t0 = esp_timer_get_time();
    for (int i = 0; i < n; i++) gpio_set_level(TEST_GPIO, i & 1);
    int64_t dt = esp_timer_get_time() - t0;
    emit("gpio", "gpio_set_level", 1, n / (dt / 1e6) / 1e6, "Mtoggle/s");
    t0 = esp_timer_get_time();
    for (int i = 0; i < n; i++) { GPIO.out_w1ts = 1u << TEST_GPIO; GPIO.out_w1tc = 1u << TEST_GPIO; }
    dt = esp_timer_get_time() - t0;
    emit("gpio", "direct_register", 1, n / (dt / 1e6) / 1e6, "Mpulse/s");
    gpio_set_level(TEST_GPIO, 0);
}

void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(500));
    uint32_t cpu_hz = 0;
    esp_clk_tree_src_get_freq_hz(SOC_MOD_CLK_CPU, ESP_CLK_TREE_SRC_FREQ_PRECISION_CACHED, &cpu_hz);
    printf("BENCH,info,cpu_mhz,0,%u,MHz\n", (unsigned)(cpu_hz / 1000000));
    printf("BENCH,info,internal_free,0,%u,bytes\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    printf("BENCH,info,psram_free,0,%u,bytes\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    bench_ram("internal", MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT, 128 * 1024, 128 * 1024);
    bench_ram("psram", MALLOC_CAP_SPIRAM, 4 * 1024 * 1024, 4 * 1024 * 1024);
    bench_flash();
    bench_gpio();
    printf("BENCH,done,done,0,0,\n");
    for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}
