#!/bin/bash
# Simple script to switch WebArena configuration
# Usage: ./switch_config.sh <config_dir>
# Example: ./switch_config.sh config_files
#          ./switch_config.sh config_files_lite

if [ $# -eq 0 ]; then
    echo "Usage: ./switch_config.sh <config_dir>"
    echo ""
    echo "Examples:"
    echo "  ./switch_config.sh config_files       # Use full WebArena (812 tasks)"
    echo "  ./switch_config.sh config_files_lite  # Use WebArena-Lite (165 tasks)"
    exit 1
fi

CONFIG_DIR="$1"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "❌ Error: Directory '$CONFIG_DIR' not found"
    exit 1
fi

echo "🔧 Switching to config: $CONFIG_DIR"
echo ""

# Find webarena package location
WEBARENA_PKG=$(python -c "import importlib.util; spec = importlib.util.find_spec('webarena'); print(spec.submodule_search_locations[0] if spec and spec.submodule_search_locations else '')" 2>/dev/null)

if [ -z "$WEBARENA_PKG" ]; then
    echo "❌ Could not find webarena package"
    echo "   Make sure you've activated the correct conda environment:"
    echo "   source /path/to/miniconda3/bin/activate rllm"
    exit 1
fi

echo "📦 Found webarena package at:"
echo "   $WEBARENA_PKG"
echo ""

# Generate combined config if needed
COMBINED_CONFIG="$CONFIG_DIR/test.raw.json"
#
# Always rebuild combined config to include latest edits
if [ -f "$COMBINED_CONFIG" ]; then
    rm -f "$COMBINED_CONFIG"
fi
if [ ! -f "$COMBINED_CONFIG" ]; then
    echo "🔧 Generating combined config file..."
    python << EOFPYTHON
import json
from pathlib import Path

config_dir = Path("$CONFIG_DIR")
config_files = sorted([f for f in config_dir.glob("*.json") if f.stem.isdigit()], key=lambda x: int(x.stem))

all_configs = []
for config_file in config_files:
    try:
        with open(config_file) as f:
            all_configs.append(json.load(f))
    except Exception as e:
        print(f"⚠️  Warning: {config_file}: {e}")

with open("$COMBINED_CONFIG", 'w') as f:
    json.dump(all_configs, f, indent=2)

print(f"✅ Created combined config with {len(all_configs)} tasks")
EOFPYTHON
    echo ""
fi

# Backup original config if not already backed up
BACKUP_FILE="$WEBARENA_PKG/test.raw.json.backup"
if [ -f "$WEBARENA_PKG/test.raw.json" ] && [ ! -f "$BACKUP_FILE" ]; then
    echo "💾 Backing up original config to:"
    echo "   $BACKUP_FILE"
    cp "$WEBARENA_PKG/test.raw.json" "$BACKUP_FILE"
    echo ""
fi

# Copy config to webarena package
echo "📋 Copying config to webarena package..."
# Remove target first to avoid any copy safeguards/alias behavior
if [ -f "$WEBARENA_PKG/test.raw.json" ]; then
    rm -f "$WEBARENA_PKG/test.raw.json"
fi
cp "$COMBINED_CONFIG" "$WEBARENA_PKG/test.raw.json"

# Verify the copy
if [ $? -eq 0 ]; then
    echo "✅ Configuration switched successfully!"
    echo ""
    echo "Summary:"
    echo "  - Config directory: $CONFIG_DIR"
    echo "  - Total tasks: $(python -c "import json; print(len(json.load(open('$COMBINED_CONFIG'))))" 2>/dev/null || echo "unknown")"

    # Show task 1 info
    TASK1_INFO=$(python -c "import json; configs = json.load(open('$COMBINED_CONFIG')); task1 = [c for c in configs if c.get('task_id') == 1]; print(task1[0]['intent'][:80] + '...' if task1 and len(task1[0]['intent']) > 80 else task1[0]['intent']) if task1 else print('Task 1 not found')" 2>/dev/null)
    if [ -n "$TASK1_INFO" ]; then
        echo "  - Task 1: $TASK1_INFO"
    fi

    echo ""
    echo "🚀 You can now run:"
    echo "   python run.py --task_name webarena.1 --max_steps 10"
    echo ""
else
    echo "❌ Failed to copy configuration"
    exit 1
fi
