import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .logger import log_error, log_info
from .models import LLMFactory


@dataclass
class BenchmarkMetrics:
    """Data class for storing benchmark metrics."""

    # Throughput metrics
    requests_per_second: float
    tokens_per_second: float

    # Latency metrics
    mean_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float

    # Time to first token metrics
    mean_ttft: float
    median_ttft: float
    p95_ttft: float
    p99_ttft: float

    # Time per output token metrics
    mean_tpot: float
    median_tpot: float
    p95_tpot: float
    p99_tpot: float

    # Overall metrics
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    total_duration: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "throughput": {
                "requests_per_second": self.requests_per_second,
                "tokens_per_second": self.tokens_per_second,
            },
            "latency": {
                "mean": self.mean_latency,
                "median": self.median_latency,
                "p95": self.p95_latency,
                "p99": self.p99_latency,
            },
            "time_to_first_token": {
                "mean": self.mean_ttft,
                "median": self.median_ttft,
                "p95": self.p95_ttft,
                "p99": self.p99_ttft,
            },
            "time_per_output_token": {
                "mean": self.mean_tpot,
                "median": self.median_tpot,
                "p95": self.p95_tpot,
                "p99": self.p99_tpot,
            },
            "summary": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "total_tokens": self.total_tokens,
                "total_duration": self.total_duration,
            },
        }


@dataclass
class RequestResult:
    """Data class for storing individual request results."""

    success: bool
    latency: float
    ttft: Optional[float]  # Time to first token
    tpot: Optional[float]  # Time per output token
    tokens: int
    error: Optional[str] = None


class BenchmarkRunner:
    """Main class for running LLM serving benchmarks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = None
        self._setup_llm()

    def _setup_llm(self):
        """Initialize the LLM instance."""
        llm_type = self.config["llm_api"]["type"]
        self.llm = LLMFactory.create_llm(llm_type, self.config)

    def run_benchmark(
        self,
        prompt: str,
        num_requests: int = 100,
        concurrent_requests: int = 10,
        request_rate: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> BenchmarkMetrics:
        """
        Run benchmark with specified parameters.

        Args:
            prompt: The prompt text to send
            num_requests: Total number of requests to send
            concurrent_requests: Maximum concurrent requests
            request_rate: Requests per second (if None, no rate limiting)
            duration: Maximum duration in seconds (if None, run until num_requests)
        """
        log_info(
            f"Starting benchmark with {num_requests} requests, {concurrent_requests} concurrent"
        )

        start_time = time.time()
        results = []

        if request_rate:
            # Rate-limited benchmark
            results = self._run_rate_limited_benchmark(
                prompt, num_requests, concurrent_requests, request_rate, duration
            )
        else:
            # Burst benchmark
            results = self._run_burst_benchmark(prompt, num_requests, concurrent_requests, duration)

        end_time = time.time()
        total_duration = end_time - start_time

        return self._calculate_metrics(results, total_duration)

    def _run_burst_benchmark(
        self, prompt: str, num_requests: int, concurrent_requests: int, duration: Optional[float]
    ) -> List[RequestResult]:
        """Run benchmark by sending requests as fast as possible."""
        results = []
        completed_requests = 0
        start_time = time.time()

        def worker():
            nonlocal completed_requests
            while True:
                # Check termination conditions
                if duration and (time.time() - start_time) >= duration:
                    break
                if completed_requests >= num_requests:
                    break

                completed_requests += 1
                if completed_requests > num_requests:
                    break

                result = self._send_request(prompt)
                results.append(result)

        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = []
            for _ in range(concurrent_requests):
                futures.append(executor.submit(worker))

            # Wait for all workers to complete
            for future in futures:
                future.result()

        return results

    def _run_rate_limited_benchmark(
        self,
        prompt: str,
        num_requests: int,
        concurrent_requests: int,
        request_rate: float,
        duration: Optional[float],
    ) -> List[RequestResult]:
        """Run benchmark with rate limiting."""
        results = []
        semaphore = threading.Semaphore(concurrent_requests)

        def send_with_semaphore():
            with semaphore:
                result = self._send_request(prompt)
                results.append(result)

        start_time = time.time()
        request_interval = 1.0 / request_rate

        with ThreadPoolExecutor(max_workers=concurrent_requests * 2) as executor:
            for i in range(num_requests):
                if duration and (time.time() - start_time) >= duration:
                    break

                executor.submit(send_with_semaphore)

                # Rate limiting - sleep until next request should be sent
                if i < num_requests - 1:  # Don't sleep after last request
                    time.sleep(request_interval)

        return results

    def _send_request(self, prompt: str) -> RequestResult:
        """Send a single request and measure performance."""
        request_start = time.time()
        ttft = None
        tpot = None
        tokens = 0

        try:
            # Simulate streaming by measuring first token time
            # In a real implementation, this would use streaming API
            first_token_start = time.time()

            response = self.llm.call(prompt)

            first_token_end = time.time()
            ttft = first_token_end - first_token_start

            # Calculate tokens and TPOT
            if response.usage:
                tokens = response.usage.completion_tokens
                if tokens > 1:
                    generation_time = time.time() - first_token_end
                    tpot = generation_time / (tokens - 1) if tokens > 1 else 0

            request_end = time.time()
            latency = request_end - request_start

            return RequestResult(success=True, latency=latency, ttft=ttft, tpot=tpot, tokens=tokens)

        except Exception as e:
            request_end = time.time()
            latency = request_end - request_start

            log_error(f"Request failed: {e}")
            return RequestResult(
                success=False, latency=latency, ttft=None, tpot=None, tokens=0, error=str(e)
            )

    def _calculate_metrics(
        self, results: List[RequestResult], total_duration: float
    ) -> BenchmarkMetrics:
        """Calculate benchmark metrics from results."""
        if not results:
            raise ValueError("No results to calculate metrics from")

        successful_results = [r for r in results if r.success]

        # Basic counts
        total_requests = len(results)
        successful_requests = len(successful_results)
        failed_requests = total_requests - successful_requests
        total_tokens = sum(r.tokens for r in successful_results)

        if not successful_results:
            log_error("All requests failed, cannot calculate meaningful metrics")
            return BenchmarkMetrics(
                requests_per_second=0,
                tokens_per_second=0,
                mean_latency=0,
                median_latency=0,
                p95_latency=0,
                p99_latency=0,
                mean_ttft=0,
                median_ttft=0,
                p95_ttft=0,
                p99_ttft=0,
                mean_tpot=0,
                median_tpot=0,
                p95_tpot=0,
                p99_tpot=0,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                total_tokens=total_tokens,
                total_duration=total_duration,
            )

        # Throughput metrics
        requests_per_second = successful_requests / total_duration
        tokens_per_second = total_tokens / total_duration

        # Latency metrics
        latencies = [r.latency for r in successful_results]
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = self._percentile(latencies, 95)
        p99_latency = self._percentile(latencies, 99)

        # TTFT metrics
        ttfts = [r.ttft for r in successful_results if r.ttft is not None]
        if ttfts:
            mean_ttft = statistics.mean(ttfts)
            median_ttft = statistics.median(ttfts)
            p95_ttft = self._percentile(ttfts, 95)
            p99_ttft = self._percentile(ttfts, 99)
        else:
            mean_ttft = median_ttft = p95_ttft = p99_ttft = 0

        # TPOT metrics
        tpots = [r.tpot for r in successful_results if r.tpot is not None]
        if tpots:
            mean_tpot = statistics.mean(tpots)
            median_tpot = statistics.median(tpots)
            p95_tpot = self._percentile(tpots, 95)
            p99_tpot = self._percentile(tpots, 99)
        else:
            mean_tpot = median_tpot = p95_tpot = p99_tpot = 0

        return BenchmarkMetrics(
            requests_per_second=requests_per_second,
            tokens_per_second=tokens_per_second,
            mean_latency=mean_latency,
            median_latency=median_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            mean_ttft=mean_ttft,
            median_ttft=median_ttft,
            p95_ttft=p95_ttft,
            p99_ttft=p99_ttft,
            mean_tpot=mean_tpot,
            median_tpot=median_tpot,
            p95_tpot=p95_tpot,
            p99_tpot=p99_tpot,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_tokens=total_tokens,
            total_duration=total_duration,
        )

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]


def save_benchmark_results(metrics: BenchmarkMetrics, output_path: str, config: Dict[str, Any]):
    """Save benchmark results to JSON file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    results = {"timestamp": timestamp, "config": config, "metrics": metrics.to_dict()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    log_info(f"Benchmark results saved to: {output_path}")


def print_benchmark_summary(metrics: BenchmarkMetrics):
    """Print benchmark results summary to console."""
    print("\n" + "=" * 60)
    print("🚀 BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n📊 THROUGHPUT METRICS:")
    print(f"  Requests/sec: {metrics.requests_per_second:.2f}")
    print(f"  Tokens/sec:   {metrics.tokens_per_second:.2f}")

    print(f"\n⏱️  LATENCY METRICS (seconds):")
    print(f"  Mean:    {metrics.mean_latency:.3f}")
    print(f"  Median:  {metrics.median_latency:.3f}")
    print(f"  P95:     {metrics.p95_latency:.3f}")
    print(f"  P99:     {metrics.p99_latency:.3f}")

    print(f"\n🎯 TIME TO FIRST TOKEN (seconds):")
    print(f"  Mean:    {metrics.mean_ttft:.3f}")
    print(f"  Median:  {metrics.median_ttft:.3f}")
    print(f"  P95:     {metrics.p95_ttft:.3f}")
    print(f"  P99:     {metrics.p99_ttft:.3f}")

    print(f"\n🔄 TIME PER OUTPUT TOKEN (seconds):")
    print(f"  Mean:    {metrics.mean_tpot:.3f}")
    print(f"  Median:  {metrics.median_tpot:.3f}")
    print(f"  P95:     {metrics.p95_tpot:.3f}")
    print(f"  P99:     {metrics.p99_tpot:.3f}")

    print(f"\n📈 SUMMARY:")
    print(f"  Total requests:      {metrics.total_requests}")
    print(f"  Successful requests: {metrics.successful_requests}")
    print(f"  Failed requests:     {metrics.failed_requests}")
    print(f"  Total tokens:        {metrics.total_tokens}")
    print(f"  Total duration:      {metrics.total_duration:.2f}s")

    success_rate = (metrics.successful_requests / metrics.total_requests) * 100
    print(f"  Success rate:        {success_rate:.1f}%")

    print("=" * 60)
