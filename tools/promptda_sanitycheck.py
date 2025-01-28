from monopriors.apis.promptda_sanity_check import SanityConfig, sanity_check
import tyro


if __name__ == "__main__":
    sanity_check(tyro.cli(SanityConfig))
