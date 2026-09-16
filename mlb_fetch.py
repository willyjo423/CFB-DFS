name: MLB fetch history

on:
  workflow_dispatch:
    inputs:
      seasons:
        description: "Seasons to fetch, space separated"
        default: "2025 2026"
      refresh:
        description: "Seasons to re-fetch even if already cached"
        default: ""
      limit:
        description: "Cap games per season (0 = all). Use 50 for a smoke test."
        default: "0"

permissions:
  contents: write

jobs:
  fetch:
    runs-on: ubuntu-latest
    timeout-minutes: 300

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Make the engine importable
        run: |
          echo "--- repo root ---"
          ls -la
          echo "--- engine/ ---"
          ls -la engine 2>/dev/null || echo "(no engine directory)"

          ENGINE_FILES="spec.py frame.py features.py model.py grade.py factors.py cache.py simulate.py ownership.py optimise.py"

          mkdir -p engine
          for f in $ENGINE_FILES; do
            if [ -f "$f" ] && [ ! -f "engine/$f" ]; then
              echo "  moving stray $f -> engine/$f"
              mv "$f" "engine/$f"
            fi
          done

          if [ ! -f engine/__init__.py ]; then
            echo "  creating engine/__init__.py"
            printf 'from .spec import SportSpec\nfrom . import frame, features\n\n__all__ = ["SportSpec", "frame", "features"]\n' > engine/__init__.py
          fi

          missing=""
          for f in $ENGINE_FILES; do
            [ -f "engine/$f" ] || missing="$missing $f"
          done
          if [ -n "$missing" ]; then
            echo ""
            echo "FATAL: these engine files are nowhere in this repo:$missing"
            echo "Upload them into an engine/ folder and re-run."
            exit 1
          fi

          python -c "import engine, engine.cache, engine.grade; print('engine imports OK')"

      - name: Commit the repair, if there was one
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -A engine
          if git diff --cached --quiet; then
            echo "engine/ was already correct"
          else
            git commit -m "Put the engine files where python can import them"
            git push
          fi

      - name: Fetch
        run: |
          python mlb_fetch.py \
            --seasons "${{ inputs.seasons }}" \
            --refresh "${{ inputs.refresh }}" \
            --limit "${{ inputs.limit }}"

      - name: Commit the cache
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          if [ -d data ]; then
            git add -f data
          fi
          if git diff --cached --quiet; then
            echo "nothing new to commit"
          else
            git commit -m "MLB history cache: ${{ inputs.seasons }}"
            git pull --rebase
            git push
          fi

      - name: What is on disk now
        if: always()
        run: ls -la data 2>/dev/null || echo "(no data directory)"
