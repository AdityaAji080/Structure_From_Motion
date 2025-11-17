from Pair_Class import Pair


def build_pairs(all_matches: list[list], min_matches: int = 8) -> list[Pair]:
    """Build list of Pair objects that have at least min_matches correspondences."""
    nimages = len(all_matches)
    pairs: list[Pair] = []

    for i in range(nimages):
        for j in range(i + 1, nimages):
            matches = all_matches[i][j]
            if matches is None or len(matches) < min_matches:
                continue
            pairs.append(Pair(cams=(i, j), matches=matches))

    print(f"Number of candidate pairs: {len(pairs)}")
    return pairs

