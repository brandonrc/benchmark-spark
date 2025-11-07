# Contributing to DGX Spark Benchmarking

Thank you for your interest in contributing! This project aims to provide reproducible, scientifically rigorous benchmarks for GPU performance analysis.

## Ways to Contribute

### 🐛 Report Issues
- Performance anomalies you've discovered
- Bugs in scripts or analysis
- Documentation improvements
- Hardware compatibility problems

### 💡 Suggest Enhancements
- New benchmark workloads
- Additional metrics to collect
- Analysis improvements
- Optimization suggestions

### 🔧 Submit Code
- Bug fixes
- New features
- Performance improvements
- Documentation updates

### 📊 Share Results
- Benchmark results from your hardware
- Comparative analysis
- Optimization findings

## Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/benchmark-spark.git
   cd benchmark-spark
   ```
3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**
5. **Test thoroughly**
6. **Submit a pull request**

## Code Guidelines

### Python Code
- Follow PEP 8 style guide
- Add docstrings to functions
- Include type hints where appropriate
- Keep functions focused and testable

### Bash Scripts
- Use `set -e` for error handling
- Add comments for complex logic
- Use meaningful variable names
- Include usage examples

### Documentation
- Use clear, concise language
- Include examples
- Keep README up to date
- Add inline comments for complex code

## Testing Checklist

Before submitting a PR:

- [ ] Code runs without errors
- [ ] Scripts have execute permissions (`chmod +x`)
- [ ] Documentation updated (if applicable)
- [ ] New features have usage examples
- [ ] No hardcoded paths or credentials
- [ ] Results are reproducible

## Benchmark Contributions

If contributing new benchmarks:

1. **Follow existing patterns**
   - Use same CSV format
   - Include warmup iterations
   - Monitor GPU metrics
   - Calculate statistics

2. **Document methodology**
   - Explain what is being measured
   - Justify test parameters
   - Describe expected outcomes

3. **Ensure reproducibility**
   - Fixed random seeds
   - Configuration files
   - Environment documentation

## Results Contributions

When sharing benchmark results:

1. **Include system specs**
   - GPU model and memory
   - CUDA and driver versions
   - Container image digest
   - CPU and RAM specs

2. **Provide raw data**
   - CSV files
   - GPU metrics logs
   - Analysis outputs

3. **Document anomalies**
   - Unexpected behavior
   - Error messages
   - Performance outliers

## Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Follow commit message conventions**:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `perf:` for performance improvements
4. **Request review** from maintainers
5. **Address feedback** promptly

## Code of Conduct

- Be respectful and constructive
- Focus on technical merit
- Help others learn
- Give credit where due

## Questions?

- Open an issue for clarification
- Check existing documentation
- Review closed issues for similar questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping improve GPU benchmarking! 🚀
