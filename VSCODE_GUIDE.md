# VS Code Configuration Guide

This project includes comprehensive VS Code configurations for easier development and debugging.

## 📁 Configuration Files

### 1. `.vscode/launch.json` - Debug Configurations

Launch configurations for running and debugging Python scripts:

#### Available Configurations:

1. **Python: Flask API** - Run the Flask API server with debugging
2. **Python: Current File** - Debug the currently open Python file
3. **Python: Test Matcher** - Run the matching test script
4. **Python: Extract Embeddings** - Run embedding extraction
5. **Python: Build Index & DB** - Build FAISS index and database
6. **Python: Interactive Demo** - Run the interactive matching demo

#### Usage:

1. Open the Debug panel (Cmd+Shift+D or Ctrl+Shift+D)
2. Select a configuration from the dropdown
3. Press F5 to start debugging

### 2. `.vscode/tasks.json` - Task Runner

Automated tasks for common operations:

#### Available Tasks:

1. **Run Flask API** - Start the Flask server
2. **Test Matching System** - Run test_matcher.py
3. **Test API Endpoints** - Run test_api.sh
4. **Extract Embeddings** - Run embedding extraction
5. **Build FAISS Index** - Build index and database
6. **Generate Crops from YOLO** - Run crop generation
7. **Train YOLO Detector** - Train the detector
8. **Run All Tests** - Execute both matcher and API tests
9. **Check Database** - Query database statistics
10. **Verify System Status** - Check index, DB, and embeddings

#### Usage:

1. Press Cmd+Shift+P (or Ctrl+Shift+P) → "Tasks: Run Task"
2. Select a task from the list
3. Or use Cmd+Shift+B (Ctrl+Shift+B) for the default build task

### 3. `.vscode/settings.json` - Workspace Settings

Project-specific VS Code settings:

#### Key Settings:

- **Python Interpreter**: `.venv/bin/python`
- **Auto-formatting**: Black (line length 100)
- **Linting**: Flake8 enabled
- **Auto-save**: After 1 second delay
- **Environment Variables**: PYTHONPATH and KMP_DUPLICATE_LIB_OK set automatically

### 4. `.vscode/extensions.json` - Recommended Extensions

Suggested extensions for this project:

1. **ms-python.python** - Python language support
2. **ms-python.vscode-pylance** - Fast Python language server
3. **ms-python.black-formatter** - Black code formatter
4. **ms-toolsai.jupyter** - Jupyter notebook support
5. **esbenp.prettier-vscode** - JSON/YAML formatter
6. **redhat.vscode-yaml** - YAML language support
7. **mtxr.sqltools** & **mtxr.sqltools-driver-sqlite** - SQLite database browser
8. **humao.rest-client** - Test API endpoints directly in VS Code

VS Code will prompt you to install these when you open the project.

### 5. `.vscode/api_tests.http` - REST Client Tests

HTTP file for testing API endpoints with the REST Client extension.

#### Usage:

1. Install REST Client extension
2. Open `api_tests.http`
3. Click "Send Request" above any HTTP request
4. View response in the panel

#### Available Tests:

- Health check
- System statistics
- Get zebra metadata (various IDs)
- Identify zebra (match testing)
- Error cases (400, 404, 501)
- Performance tests
- Batch identify requests

## 🚀 Quick Start

### Run Flask API with Debugging

1. Open VS Code
2. Press F5
3. Select "Python: Flask API"
4. Set breakpoints as needed
5. API starts on http://127.0.0.1:5001

### Test API with REST Client

1. Open `.vscode/api_tests.http`
2. Click "Send Request" above any request
3. View formatted JSON response

### Run Tests

#### Option 1: Using Tasks

1. Cmd+Shift+P → "Tasks: Run Task"
2. Select "Run All Tests"

#### Option 2: Using Terminal

```bash
python test_matcher.py
./test_api.sh
```

### Check System Status

1. Cmd+Shift+P → "Tasks: Run Task"
2. Select "Verify System Status"

Or run directly:

```bash
python -c "import faiss, sqlite3; ..."
```

## 🔧 Customization

### Modify Python Interpreter

Edit `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

### Add New Debug Configuration

Edit `.vscode/launch.json`:

```json
{
  "name": "My Custom Script",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/my_script.py",
  "console": "integratedTerminal"
}
```

### Add New Task

Edit `.vscode/tasks.json`:

```json
{
  "label": "My Task",
  "type": "shell",
  "command": "${workspaceFolder}/.venv/bin/python",
  "args": ["my_script.py"]
}
```

## 🐛 Debugging Tips

### Debug Flask API

1. Set breakpoints in `scripts/app.py`
2. Press F5 and select "Python: Flask API"
3. Make API requests with curl or REST Client
4. Execution pauses at breakpoints

### Debug Matching Logic

1. Set breakpoints in `test_matcher.py` or `examples/match_demo.py`
2. Use "Python: Test Matcher" or "Python: Interactive Demo"
3. Step through code with F10 (Step Over), F11 (Step Into)

### View Variables

- Hover over variables while debugging
- Use Debug Console (Cmd+Shift+Y) to evaluate expressions
- Watch window for monitoring specific variables

## 📊 Database Browser

### Using SQLTools Extension

1. Install SQLTools extensions (recommended in extensions.json)
2. Click SQLTools icon in Activity Bar
3. Add Connection:
   - Driver: SQLite
   - Database: `${workspaceFolder}/zebra.db`
4. Browse tables, run queries directly in VS Code

### Query Examples

```sql
-- Total crops
SELECT COUNT(*) FROM crops;

-- High-confidence detections
SELECT crop_path, confidence
FROM crops
WHERE confidence > 0.95
LIMIT 10;

-- Get specific zebra
SELECT * FROM crops WHERE faiss_id = 0;
```

## 🧪 Testing Workflow

### Complete Test Cycle

1. **Verify System**

   - Task: "Verify System Status"
   - Checks index, database, embeddings

2. **Test Matching**

   - Task: "Test Matching System"
   - Runs test_matcher.py

3. **Test API**

   - Start API: F5 → "Python: Flask API"
   - Run tests: Task → "Test API Endpoints"
   - Or use REST Client: `.vscode/api_tests.http`

4. **Check Database**
   - Task: "Check Database"
   - Or use SQLTools browser

## 🎯 Keyboard Shortcuts

| Action            | Shortcut (Mac) | Shortcut (Windows/Linux)   |
| ----------------- | -------------- | -------------------------- |
| Start Debugging   | F5             | F5                         |
| Run Task          | Cmd+Shift+B    | Ctrl+Shift+B               |
| Command Palette   | Cmd+Shift+P    | Ctrl+Shift+P               |
| Toggle Terminal   | Ctrl+`         | Ctrl+`                     |
| Debug Console     | Cmd+Shift+Y    | Ctrl+Shift+Y               |
| Send HTTP Request | Cmd+Alt+R      | Ctrl+Alt+R (in .http file) |

## 📝 Environment Variables

Automatically set in terminal (via settings.json):

```bash
PYTHONPATH=${workspaceFolder}
KMP_DUPLICATE_LIB_OK=TRUE  # Allows FAISS + PyTorch coexistence
```

## 🔗 Useful Extensions

### Python Development

- **Pylance**: Fast, feature-rich Python IntelliSense
- **Black Formatter**: Automatic code formatting
- **Python**: Core Python support

### API Testing

- **REST Client**: Test HTTP endpoints in .http files
- **Thunder Client**: Alternative REST API client

### Database

- **SQLTools**: Browse and query SQLite database
- **SQLite Viewer**: View .db files

### General

- **Prettier**: Format JSON, YAML
- **YAML**: YAML syntax support
- **GitLens**: Enhanced Git integration

## 💡 Tips & Tricks

### 1. Quick API Testing

Open `api_tests.http` and click "Send Request" - no need for Postman!

### 2. Auto-format on Save

Code automatically formats with Black when you save (Cmd+S)

### 3. Environment Activation

Terminal automatically activates .venv when opened

### 4. Integrated Debugging

Set breakpoints, inspect variables, step through code - all in VS Code

### 5. Task Chaining

Create task dependencies to run multiple tasks in sequence

## 🆘 Troubleshooting

### Python Interpreter Not Found

1. Cmd+Shift+P → "Python: Select Interpreter"
2. Choose `.venv/bin/python`

### Tasks Not Working

1. Ensure `.venv` is activated
2. Check file permissions: `chmod +x test_api.sh`
3. Verify paths in tasks.json

### REST Client Not Working

1. Install "REST Client" extension
2. Ensure Flask API is running
3. Check baseUrl in api_tests.http

### Debug Not Starting

1. Check launch.json configuration
2. Verify Python extension is installed
3. Ensure file paths are correct

## 📚 Additional Resources

- [VS Code Python Debugging](https://code.visualstudio.com/docs/python/debugging)
- [VS Code Tasks](https://code.visualstudio.com/docs/editor/tasks)
- [REST Client Extension](https://marketplace.visualstudio.com/items?itemName=humao.rest-client)
- [SQLTools Extension](https://marketplace.visualstudio.com/items?itemName=mtxr.sqltools)

---

**All configurations are ready to use!** Just open the project in VS Code and start coding. 🚀
