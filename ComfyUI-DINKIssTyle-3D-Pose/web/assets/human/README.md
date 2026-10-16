# Human base mesh

This is prepared graphical data from the official MakeHuman project, released
under CC0-1.0. It contains no MakeHuman application or shader code.

Sources downloaded on 2026-10-15:

- [Body mesh](https://github.com/makehumancommunity/makehuman/blob/master/makehuman/data/3dobjs/base.obj)
- [Default skeleton](https://github.com/makehumancommunity/makehuman/blob/master/makehuman/data/rigs/default.mhskel)
- [Authored skin weights](https://github.com/makehumancommunity/makehuman/blob/master/makehuman/data/rigs/default_weights.mhw)
- [Asset license](https://github.com/makehumancommunity/makehuman/blob/master/LICENSE.ASSETS.md)
- [License clarification](https://static.makehumancommunity.org/about/license.html)

Source copyright holders: Data Collection AB, Joel Palmius, Jonas Hauquier.
The license is included in LICENSE.CC0.md; source SHA-256 checksums and counts
are included in provenance.json.

Preparation uses tools/build_human_asset.py. Only the body face group is kept;
helper meshes are excluded. Original quads and authored weights are retained,
with weights mapped onto the editor's 52 controls. The browser applies one
Catmull-Clark subdivision step to the model and weights. The bone bind pose is
built from authored joint positions, then posed through the editor controls.
Body and head dimensions are adjusted for the selected head proportion.

All files are local. No runtime model download or Blender installation is needed.
