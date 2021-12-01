Examples might need to import modules from other packages, e.g., `src/`.

One possibility to run examples is to append the root folder of the repository to the `PYTHONPATH` environment variable.
For example:
1. `cd` your terminal to the root folder of the repository
1. Run the desired example as follows
```bash
PYTHONPATH=$PYTHONPATH:. python examples/your-example.py
```

A similar result can be obtained by exporting the updated `PYTHONPATH` to each newly opened terminal:
1. `cd` your terminal to the root folder of the repository
1. Add the current folder to the `PYTHONPATH`
```bash
export PYTHONPATH=$PYTHONPATH:.
```
1. Run the desired example as follows
```bash
python examples/your-example.py
```

A further possibility is to setup your IDE to automatically export the root folder to the `PYTHONPATH`.
For example, in VS Code it suffices to add `"env": {"PYTHONPATH": "${PYTHONPATH}:${workspaceFolder}"}` in the configuration contained in `.vscode/launch.json`.
In some VS Code releases, it is also necessary to set `"terminal.integrated.env.<platform>": {"PYTHONPATH": "${workspaceFolder}"}` (where `<platform>` might be `windows`,`linux` or `osx`) in the settings file contained in `.vscode/settings.json`.