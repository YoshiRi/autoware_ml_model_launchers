# AGENTS.md

## Branch policy

- Never push directly to `main`.
- Treat `main` as the production/stable branch.
- Use `dev` as the integration branch.
- Start all implementation work from the latest `dev`.
- For each task, create a branch from `dev`:
  - `feature/<short-description>` for new features
  - `fix/<short-description>` for bug fixes
  - `refactor/<short-description>` for refactoring
  - `docs/<short-description>` for documentation
- Open pull requests into `dev`, not `main`.
- Do not merge your own pull request unless explicitly instructed.
- If `dev` has moved, rebase or merge the latest `dev` before finalizing the pull request.
- Keep pull requests small and focused.
- Promote tested releases from `dev` to `main` with a separate pull request.

## Workflow

Before changing code:

1. Check the current branch and working tree:

   ```bash
   git status --short --branch
   ```

2. Update `dev` and create a task branch:

   ```bash
   git fetch origin
   git switch dev
   git pull --ff-only origin dev
   git switch -c feature/<short-description>
   ```

3. Inspect the relevant files and existing conventions.
4. Make the minimal necessary changes.

Before opening or finalizing a pull request:

1. Run the formatting and whitespace checks.
2. Run lint.
3. Build and run tests from the Autoware workspace root.
4. Update documentation when behavior changes.
5. Rebase on the latest `dev` when necessary:

   ```bash
   git fetch origin
   git rebase origin/dev
   ```

6. Summarize:
   - what changed
   - why it changed
   - how it was tested
   - risks and follow-up tasks

## Commands

Run formatting and static checks from this repository:

```bash
git diff --check
bash -n scripts/check_environment
xmllint --noout package.xml launch/*.launch.xml
source /opt/ros/humble/setup.bash
python3 -m ament_flake8.main \
  --config setup.cfg \
  autoware_ml_model_launchers setup.py
python3 -m ament_pep257.main autoware_ml_model_launchers
```

Build and test from the Autoware workspace root:

```bash
cd ~/autoware
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select autoware_ml_model_launchers
colcon test --packages-select autoware_ml_model_launchers
colcon test-result --verbose
```

For Open YOLO runtime checks, activate its virtual environment before sourcing the workspace:

```bash
source ~/venvs/open_yolo/bin/activate
source ~/autoware/install/setup.bash
ros2 run autoware_ml_model_launchers check_environment \
  --pipeline open_yolo \
  --camera camera5
```
