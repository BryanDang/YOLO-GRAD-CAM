# GitHub Actions Workflows

This directory contains automated workflows for maintaining the YoloCAM library with minimal manual intervention.

## Workflows Overview

### 🔄 Continuous Integration (`ci.yml`)
**Triggers**: Push to main/develop, Pull requests, Weekly schedule
- **Multi-platform testing** (Ubuntu, Windows, macOS)
- **Multi-version Python support** (3.8-3.12)
- **Code quality checks** (linting, formatting, type checking)
- **Test execution** (unit and integration tests)
- **Security scanning** (safety, bandit)
- **Package building and validation**
- **Documentation building**

### 🚀 Release Automation (`release.yml`)
**Triggers**: Git tags (v*)
- **Automated testing** before release
- **Package building and publishing** to PyPI
- **GitHub release creation** with changelog
- **Documentation deployment** to GitHub Pages
- **Post-release notifications**

### 🔒 Dependency Management (`dependencies.yml`)
**Triggers**: Weekly schedule, Manual trigger
- **Automated dependency updates** with testing
- **Security vulnerability scanning** (safety, pip-audit)
- **License compliance checking**
- **Automatic PR creation** for updates
- **Security issue creation** when vulnerabilities found

### 🛠️ Code Quality & Maintenance (`maintenance.yml`)
**Triggers**: Monthly schedule, Manual trigger
- **Code complexity analysis** (radon)
- **Dead code detection** (vulture)
- **Security analysis** (bandit, semgrep)
- **Dependency health checks** (unused/missing deps)
- **Performance baseline** tracking
- **Documentation quality** validation
- **Automated issue creation** for problems

## Automation Benefits

### 🔧 **Zero-Maintenance Operation**
- **Automatic dependency updates** with safety testing
- **Continuous security monitoring** and alerting
- **Code quality regression detection**
- **Performance baseline tracking**

### 🚀 **Seamless Releases**
- **One-click releases** via git tags
- **Automatic PyPI publishing** with trusted publishing
- **Documentation deployment** to GitHub Pages
- **Changelog generation** from commit history

### 🛡️ **Proactive Security**
- **Weekly vulnerability scanning**
- **License compliance monitoring**
- **Security issue auto-creation**
- **Multiple security tools** (safety, bandit, semgrep)

### 📊 **Quality Assurance**
- **Cross-platform compatibility** testing
- **Multi-version Python** support validation
- **Code complexity** monitoring
- **Dead code** detection and cleanup

## Setup Instructions

### 1. Repository Secrets
Add these secrets to your repository for full functionality:

```bash
# Required for PyPI publishing (if not using trusted publishing)
PYPI_API_TOKEN=your_pypi_token

# Optional: For enhanced notifications
SLACK_WEBHOOK=your_slack_webhook_url
DISCORD_WEBHOOK=your_discord_webhook_url
```

### 2. PyPI Trusted Publishing (Recommended)
Configure trusted publishing on PyPI for secure, keyless publishing:
1. Go to PyPI → Manage → Publishing
2. Add GitHub as trusted publisher
3. Configure: `owner/repo`, `release.yml`, `release` environment

### 3. GitHub Pages (Optional)
Enable GitHub Pages for automatic documentation deployment:
1. Repository Settings → Pages
2. Source: GitHub Actions
3. Domain: Configure custom domain if desired

### 4. Branch Protection
Configure branch protection rules:
```yaml
Branch: main
Rules:
  - Require status checks (CI workflow)
  - Require up-to-date branches
  - Restrict pushes to specific users/teams
```

## Workflow Customization

### Adding New Checks
To add new quality checks, modify `maintenance.yml`:
```yaml
- name: Custom Quality Check
  run: |
    # Your custom quality tool
    your-tool src/yolocam >> quality-report.md
```

### Adjusting Schedules
Modify cron expressions in workflow triggers:
```yaml
schedule:
  - cron: '0 6 * * 1'  # Weekly on Monday 6 AM UTC
  - cron: '0 6 1 * *'  # Monthly on 1st at 6 AM UTC
```

### Adding Notifications
Extend notification steps in workflows:
```yaml
- name: Custom Notification
  uses: your-notification-action@v1
  with:
    webhook: ${{ secrets.YOUR_WEBHOOK }}
    message: "Custom notification message"
```

## Monitoring

### Workflow Status
Monitor workflow status via:
- **GitHub Actions tab** in repository
- **Status badges** in README (optional)
- **Email notifications** (GitHub settings)

### Quality Metrics
Track quality metrics through:
- **Issues created** by automation workflows
- **Workflow artifacts** with detailed reports
- **Performance benchmarks** (if enabled)
- **Security scan results**

### Dependency Health
Monitor dependency health via:
- **Automated PRs** for updates
- **Security issues** for vulnerabilities
- **License compliance** reports

## Troubleshooting

### Common Issues

**1. PyPI Publishing Fails**
```bash
# Check secrets and trusted publishing configuration
# Verify package version isn't already published
```

**2. Tests Fail on Specific Platforms**
```bash
# Check platform-specific dependencies
# Review test isolation and mocking
```

**3. Quality Checks Too Strict**
```bash
# Adjust thresholds in maintenance.yml
# Add exceptions for specific cases
```

**4. High Resource Usage**
```bash
# Reduce matrix size in ci.yml
# Optimize test execution
# Use caching more effectively
```

### Getting Help
- Check workflow logs for detailed error messages
- Review repository issues for automated reports
- Consult GitHub Actions documentation
- Check tool-specific documentation (pytest, black, etc.)

---

This automation system provides comprehensive, maintenance-free operation of your YoloCAM library with proactive monitoring and quality assurance.