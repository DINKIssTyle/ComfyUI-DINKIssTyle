import asyncio
import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class MonitorTests(unittest.TestCase):
    def setUp(self):
        server = MagicMock()
        server.PromptServer.instance.routes.get.side_effect = lambda path: lambda fn: fn
        self.psutil = MagicMock()
        self.psutil.cpu_percent.return_value = 25
        self.psutil.virtual_memory.return_value = type('RAM', (), {'total': 16000, 'available': 6000})()
        with patch.dict(sys.modules, {'server': server, 'aiohttp': MagicMock(), 'psutil': self.psutil}):
            module = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'ComfyUI-DINKIssTyle/dinki_monitor.py'))
        self.env = module['_collect'].__globals__

    def test_multiple_gpus_and_unavailable_sensor_values(self):
        parsed = self.env['_parse_gpus']('0, RTX 4090, 97, 65, 12000, 24564\n1, RTX 3090, N/A, [Not Supported], 100, 24576\n')
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]['utilization'], 97)
        self.assertEqual(parsed[0]['temperature'], 65)
        self.assertIsNone(parsed[1]['temperature'])
        self.assertIsNone(parsed[1]['utilization'])

    def test_bad_rows_and_nonfinite_numbers(self):
        self.assertEqual(self.env['_parse_gpus']('driver error\na, b, c, d, e, f\n'), [])
        for value in ('nan', 'inf', '-1', 'N/A'):
            self.assertIsNone(self.env['_number'](value))

    def test_gpu_failure_keeps_cpu_and_ram(self):
        with patch.dict(self.env, {'_nvidia_smi_path': lambda: 'nvidia-smi'}), patch.object(
            self.env['subprocess'], 'run', side_effect=subprocess.TimeoutExpired('nvidia-smi', 3)
        ):
            result = self.env['_collect']()
        self.assertEqual(result['cpu_percent'], 25)
        self.assertEqual(result['ram']['used_bytes'], 10000)
        self.assertEqual(result['gpus'], [])
        self.assertIn('timed out', result['errors'][0])

    def test_no_driver_or_psutil_is_nonfatal(self):
        with patch.dict(self.env, {'psutil': None, '_nvidia_smi_path': lambda: None}):
            result = self.env['_collect']()
        self.assertIsNone(result['cpu_percent'])
        self.assertIsNone(result['ram'])
        self.assertEqual(len(result['errors']), 2)

    def test_windows_command_has_no_console_and_a_timeout(self):
        completed = MagicMock(stdout='0, RTX 4090, 80, 60, 1024, 24576\n')
        with patch.dict(self.env, {'_nvidia_smi_path': lambda: 'nvidia-smi.exe'}), patch.object(
            self.env['os'], 'name', 'nt'
        ), patch.object(self.env['subprocess'], 'CREATE_NO_WINDOW', 0x08000000, create=True), patch.object(
            self.env['subprocess'], 'run', return_value=completed
        ) as run:
            result = self.env['_collect']()
        self.assertEqual(result['gpus'][0]['memory_used_mib'], 1024)
        self.assertEqual(run.call_args.kwargs['creationflags'], 0x08000000)
        self.assertEqual(run.call_args.kwargs['timeout'], 3)
        self.assertNotIn('shell', run.call_args.kwargs)

    def test_cache_shares_sample_and_expires(self):
        collect = MagicMock(return_value={'gpus': []})
        clock = MagicMock(return_value=10)
        with patch.dict(self.env, {'_collect': collect}), patch.object(self.env['time'], 'monotonic', clock):
            cache = self.env['MonitorCache']()
            cache.read()
            cache.read()
            self.assertEqual(collect.call_count, 1)
            clock.return_value = 12
            cache.read()
            self.assertEqual(collect.call_count, 2)

    def test_concurrent_requests_share_one_background_sample(self):
        collect = MagicMock(return_value={'gpus': []})
        async def concurrent():
            cache = self.env['MonitorCache']()
            return await asyncio.gather(*(asyncio.to_thread(cache.read) for _ in range(8)))
        with patch.dict(self.env, {'_collect': collect}):
            results = asyncio.run(concurrent())
        self.assertEqual(collect.call_count, 1)
        self.assertEqual(len(results), 8)


if __name__ == '__main__':
    unittest.main()
