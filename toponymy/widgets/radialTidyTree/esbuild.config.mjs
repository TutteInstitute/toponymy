import * as esbuild from 'esbuild';

/* Common build options ------------------------------------------------ */
const options = {
  entryPoints: ['src/widget.js'],
  outfile:     `dist/widget.js`,
  bundle:      true,
  format:      'esm',
  sourcemap:   false,
  minify:      true,
};

await esbuild.build(options);
