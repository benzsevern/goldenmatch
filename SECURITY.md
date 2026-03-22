# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| < 0.3   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in GoldenMatch, please report it responsibly:

1. **Do not** open a public issue
2. Email **benzsevern@gmail.com** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You will receive a response within 48 hours
4. A fix will be prioritized and released as a patch version

## Scope

GoldenMatch processes potentially sensitive data (PII, healthcare records, financial data). Security concerns include:

- **Data leakage** -- ensuring PII in domain packs or configs isn't exposed
- **LLM data exposure** -- pairs sent to OpenAI/Anthropic APIs for LLM scoring
- **Injection** -- YAML config parsing, SQL in database connectors
- **Bloom filter privacy** -- PPRL implementation correctness

## Best Practices for Users

- Store API keys in environment variables, not config files
- Use the `.testing/` directory (gitignored) for credentials
- Review LLM scorer config before enabling -- borderline pairs are sent to external APIs
- Use PPRL (bloom filter transforms + dice/jaccard scoring) when matching across organizations without sharing raw PII
