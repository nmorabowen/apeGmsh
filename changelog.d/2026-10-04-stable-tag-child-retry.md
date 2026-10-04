### TESTS - retry the section-tag child once on a Windows native crash (#1376)

`test_the_tag_is_the_same_under_two_hash_seeds` retries its fresh `python -c` child once when it dies with 0xC000070A and empty stdout and stderr. Any other failure, or a second crash, still fails.
