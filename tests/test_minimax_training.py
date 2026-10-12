"""H3 recipes, managed snapshots, artifacts and isolated engine checks."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from safetensors.torch import save_file

NODE = Path(__file__).resolve().parents[1] / 'ComfyUI-DINKIssTyle'
pkg = types.ModuleType('dkst_minimax_tests')
pkg.__path__ = [str(NODE)]
with patch.dict(sys.modules, {'dkst_minimax_tests':pkg}):
    config = importlib.import_module('dkst_minimax_tests.minimax_trainer.config')
    runner = importlib.import_module('dkst_minimax_tests.minimax_trainer.runner')
    jobs = importlib.import_module('dkst_minimax_tests.qwen21_trainer.jobs')


class TestMinimax(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.input = self.root / 'input'
        self.input.mkdir()
        Image.new('RGB',(96,64),'red').save(self.input/'portrait.png')
        self.service = jobs.Jobs(self.root/'h3',self.root/'loras',{'input':self.input},
            config=config,runner=NODE/'minimax_trainer/runner.py',output_prefix='DKST_MinimaxH3',label='MiniMax H3')
        self.runtime={'python':sys.executable,'turbo_lora':'','ref_dit':'','training_adapter':''}
        for name in ['dit','vae','text_encoder']:
            file=self.root/(name+'.safetensors')
            file.write_bytes(b'model fixture')
            self.runtime[name]=str(file)
        self.service.save_settings(self.runtime)
        self.data={'project_name':'person','trigger_word':'dkst_person','sample_every':0,
                   'image_gallery':[{'filename':'portrait.png','caption':'red coat'}]}

    def prepare(self):
        path=self.service.prepare(self.data)
        return path,json.loads(path.read_text())

    def test_defaults_and_no_llm_runtime_needed(self):
        path,job=self.prepare()
        self.assertEqual(job['options']['epochs'],50)
        self.assertEqual(job['options']['resolution'],512)
        self.assertEqual(job['engine_version'],config.ENGINE_VERSION)
        self.assertIn('DKST_MinimaxH3',job['output_dir'])
        self.assertNotIn('caption_model',job['runtime'])
        self.assertIn('dkst_person, red coat',next(Path(job['image_dir']).glob('*.txt')).read_text())
        self.assertNotIn('model fixture',Path(job['dataset_config']).read_text())
        argv=config.commands(job,runner.VENDOR)
        train=argv['training']
        for flag,value in [('--family','minimax'),('--network_dim','8'),('--learning_rate','1e-6'),
                           ('--optimizer_type','automagic3'),('--train_blocks','20-49')]:
            self.assertEqual(train[train.index(flag)+1],value)
        self.assertIn('preview_frames=1',train)
        self.assertIn('distill=0',train)
        self.assertNotIn('--training_adapter',train)
        self.assertNotIn('--sample_prompts',train)
        self.assertNotIn('--adaptive_lr',train)

    def test_ref2va_requires_separate_dit_before_creating_a_job(self):
        self.data['training_base']='ref2va'
        with self.assertRaisesRegex(ValueError,'ref2va DiT'):
            self.prepare()
        self.assertFalse((self.root/'h3/projects').exists())
        reference=self.root/'ref.safetensors';reference.write_bytes(b'reference')
        self.runtime['ref_dit']=str(reference)
        self.service.save_settings(self.runtime)
        _,job=self.prepare()
        argv=config.commands(job,runner.VENDOR)['training']
        self.assertEqual(argv[argv.index('--dit')+1],str(reference.resolve()))
        self.assertNotIn('distill=1',argv)

    def test_preview_adapter_standard_mode_and_resume_arguments(self):
        for key in ['training_adapter','turbo_lora']:
            file=self.root/(key+'.safetensors');file.write_bytes(b'adapter')
            self.runtime[key]=str(file)
        self.service.save_settings(self.runtime)
        self.data.update(sample_every=5,preset='Standard',training_mode='More Blocks',training_structure='Model default')
        _,job=self.prepare()
        job['resume_state']='/complete-state'
        train=config.commands(job,runner.VENDOR)['training']
        self.assertEqual(train[train.index('--network_dim')+1],'16')
        self.assertEqual(train[train.index('--train_blocks')+1],'6-49')
        self.assertEqual(train[train.index('--sample_steps')+1],'6')
        self.assertIn('lownoise_pct=8',train)
        self.assertEqual(train[train.index('--speed_lora_strength')+1],'0.75')
        self.assertIn('/complete-state',train)
        self.assertIn('--training_adapter',train)

    def test_invalid_options_rejected(self):
        for data in [{'resolution':500},{'precision':'bf16'},{'epochs':True},{'training_base':'Qwen'},
                     {'training_mode':'all'},{'sample_every':-1},{'trigger_word':''}]:
            with self.subTest(data=data),self.assertRaises(ValueError):config.training_options(data)

    def test_generated_training_argv_is_accepted_by_the_vendored_cli(self):
        import argparse
        import ast
        _,job=self.prepare()
        job['options']['sample_every']=5
        job['runtime']['training_adapter']='/adapter.safetensors'
        job['runtime']['turbo_lora']='/turbo.safetensors'
        job['resume_state']='/complete-state'
        source=ast.parse((runner.VENDOR/'src/fizgig/families/train.py').read_text())
        parser=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='setup_parser')
        scope={'argparse':argparse,'quant':types.SimpleNamespace(PRECISIONS=('bf16','int8','nf4','hqq'))}
        exec(compile(ast.Module(body=[parser],type_ignores=[]),'<Fizgig CLI>','exec'),scope)
        parsed=scope['setup_parser']().parse_args(config.commands(job,runner.VENDOR)['training'][3:])
        self.assertEqual(parsed.family,'minimax')
        self.assertEqual(parsed.resume,'/complete-state')
        self.assertEqual(parsed.sample_steps,6)

    def test_shared_coordinator_launches_h3_entry_not_qwen(self):
        process=types.SimpleNamespace(poll=lambda:None)
        with patch.object(jobs.subprocess,'Popen',return_value=process) as popen:
            state=self.service.start(self.data)
        self.assertEqual(popen.call_args.args[0][2],str(NODE/'minimax_trainer/runner.py'))
        self.assertEqual(state['status'],'starting')
        self.assertEqual(self.service.control(state['id'],'pause')['status'],'pause_requested')

    def test_h3_artifact_pair_validation_rejects_qwen(self):
        file=self.root/'lora.safetensors'
        save_file({'transformer.test.lora_A.weight':torch.ones(2,4),'transformer.test.lora_B.weight':torch.ones(4,2)},str(file))
        with self.assertRaisesRegex(RuntimeError,'H3 LoRA'):runner.validate_lora(file)
        save_file({'lora_unet_blocks_20_attn_qkv_proj.lora_down.weight':torch.ones(2,4)},str(file))
        with self.assertRaises(RuntimeError):runner.validate_lora(file)
        save_file({'lora_unet_blocks_20_attn_qkv_proj.lora_down.weight':torch.ones(2,4),
                   'lora_unet_blocks_20_attn_qkv_proj.lora_up.weight':torch.ones(4,2)},str(file))
        runner.validate_lora(file)

    def test_text_invalidation_does_not_touch_h3_latents(self):
        _,job=self.prepare()
        cache=Path(job['cache_dir']);stem=next(Path(job['image_dir']).glob('*.png')).stem
        text=cache/(stem+'_minimaxh3_te.safetensors')
        caption=next(Path(job['image_dir']).glob('*.txt')).read_text()
        save_file({'hidden_states':torch.zeros(1,5120)},str(text),metadata={'caption1':caption})
        latent=cache/'latent_minimaxh3.safetensors';latent.write_bytes(b'preserve')
        runner.coordinator.write_json(cache/'text_encoder.json',config.model_signature(job['runtime']['text_encoder']))
        runner.invalidate_text_cache(job);self.assertTrue(text.exists())
        next(Path(job['image_dir']).glob('*.txt')).write_text('new caption')
        runner.invalidate_text_cache(job);self.assertFalse(text.exists());self.assertEqual(latent.read_bytes(),b'preserve')

    def test_h3_completion_uses_h3_validation_and_vendor_in_fresh_stages(self):
        path,job=self.prepare();final=Path(job['output_dir'])/'person.safetensors'
        fixture=self.root/'valid.safetensors'
        save_file({'lora_unet_blocks_20_attn_qkv_proj.lora_down.weight':torch.ones(2,4),
                   'lora_unet_blocks_20_attn_qkv_proj.lora_up.weight':torch.ones(4,2)},str(fixture))
        argv={stage:[sys.executable,'-c',"print('cache')"] for stage in ['latents','text']}
        argv['training']=[sys.executable,'-c',f"import shutil;shutil.copyfile({str(fixture)!r},{str(final)!r});print('epoch 50/50  avr_loss=0.123  step=1050')"]
        with patch.object(runner,'check_environment',return_value={'ok':True}),patch.object(runner,'validate_models'),\
             patch.object(runner,'invalidate_text_cache',return_value={}),patch.object(runner,'commands',return_value=argv):
            runner.coordinator.run_job(path,engine=runner)
        state=self.service.status(job['id'])
        self.assertEqual(state['status'],'completed');self.assertEqual(state['lora_path'],str(final));self.assertEqual(state['epoch'],50)

    def test_actual_tiny_h3_training_in_a_clean_process(self):
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('minimax_training_smoke.py'))],capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Kohya save/load roundtrip',result.stdout)

    def test_vendor_manifest_and_only_h3_driver(self):
        import hashlib
        manifest=json.loads((runner.VENDOR/'UPSTREAM.json').read_text())
        self.assertEqual(manifest['commit'],'33454e1b0bf06dd2203551abd3bdcdfd8cdaee68')
        for name,digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((runner.VENDOR/'src'/name).read_bytes()).hexdigest(),digest,name)
        self.assertTrue((runner.VENDOR/'src/fizgig/assets/qwen3vl_tokenizer/tokenizer.json').is_file())
        self.assertFalse((runner.VENDOR/'src/fizgig/families/qwen_image.py').exists())

if __name__=='__main__':unittest.main()
