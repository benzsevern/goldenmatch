"""GoldenMatch PPRL -- Privacy-Preserving Record Linkage.

Multi-party entity resolution where no party sees the other's raw data.

Two modes:
- trusted_third_party: parties send encrypted bloom filters to coordinator
- smc: secure multi-party computation via secret sharing (requires goldenmatch[pprl])
"""
