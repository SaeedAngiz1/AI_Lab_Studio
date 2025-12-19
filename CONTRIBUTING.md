# Contributing to AI Lab Studio

Thank you for your interest in contributing to AI Lab Studio! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/SaeedAngiz1/AI_Lab_Studio/issues)
2. If not, create a new issue using the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md)
3. Provide as much detail as possible:
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Screenshots if applicable

### Suggesting Features

1. Check if the feature has already been suggested
2. Create a new issue using the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md)
3. Explain the use case and benefits

### Submitting Code Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/SaeedAngiz1/AI_Lab_Studio.git
   cd AI_Lab_Studio
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest black flake8  # Development dependencies
   ```

4. **Make your changes**
   - Write clean, readable code
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation if needed

5. **Test your changes**
   ```bash
   # Run the app locally
   streamlit run app.py
   
   # Run tests (if available)
   pytest
   
   # Check code style
   black --check .
   flake8 .
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```
   Use clear, descriptive commit messages.

7. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub using the PR template.

## Development Guidelines

### Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Keep functions focused and small
- Use descriptive variable names

### Code Formatting

We use `black` for code formatting:
```bash
black .
```

### Linting

We use `flake8` for linting:
```bash
flake8 .
```

### Project Structure

```
ai_lab_studio/
├── app.py                 # Main entry point
├── pages/                 # Streamlit pages
├── utils/                 # Utility modules
├── models/                # Saved models (gitignored)
├── sample_data/           # Sample datasets
└── tests/                 # Test files (to be added)
```

### Adding New Features

1. **New ML Algorithm**
   - Add to `utils/ml_models.py`
   - Update the model selection UI in `pages/2_ML_Training.py`
   - Add tests if possible

2. **New Preprocessing Step**
   - Add to `utils/data_processor.py`
   - Update UI in `pages/1_Data_Hub.py`
   - Document usage

3. **New Page**
   - Create file in `pages/` directory
   - Follow naming convention: `N_Page_Name.py`
   - Update main app navigation if needed

### Testing

- Write tests for new functionality
- Ensure existing tests pass
- Test with sample datasets
- Test edge cases

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update inline comments for complex logic
- Update DEPLOYMENT.md for deployment changes

## Pull Request Process

1. **Ensure your PR:**
   - Addresses an existing issue (or create one first)
   - Follows the project structure
   - Includes tests if applicable
   - Updates documentation
   - Passes all CI checks

2. **PR Title Format:**
   - `[FEATURE] Description` for new features
   - `[BUGFIX] Description` for bug fixes
   - `[DOCS] Description` for documentation
   - `[REFACTOR] Description` for refactoring

3. **PR Description:**
   - Use the PR template
   - Describe what changes were made
   - Link related issues
   - Include screenshots for UI changes

4. **Review Process:**
   - Maintainers will review your PR
   - Address any feedback or requested changes
   - Once approved, your PR will be merged

## Questions?

- Open an issue for questions
- Check existing documentation
- Review code comments

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to AI Lab Studio! 🎉

