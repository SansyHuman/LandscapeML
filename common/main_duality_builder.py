import os
import sys
from enum import Enum
import math


class LieGroup(Enum):
    SU = 1
    SO = 2
    Sp = 3


def get_seiberg_conformal_window(group: LieGroup, n_c: int) -> tuple[float, float]:
    """
    Gets the conformal window for a given Lie group and number of colors of fundamental chirals only.
    :param group: The Lie group.
    :param n_c: The number of colors.
    :return: The conformal window as a tuple of floats. The flavor should be larger
    than the first value and less than the second value.
    """
    if group == LieGroup.SU:
        return 1.5 * n_c, 3 * n_c
    elif group == LieGroup.SO:
        return 1.5 * (n_c - 2), 3 * (n_c - 2)
    elif group == LieGroup.Sp:
        return 1.5 * (n_c + 1), 3 * (n_c + 1)

    return 0, 0


if __name__ == '__main__':
    print(sys.version)
    print("GIL enabled:", sys._is_gil_enabled())

    filename = input("Enter the filename to save dual theory data: ")

    seiberg = []
    used_pair = set()

    su_min = int(input("Enter the minimum N of SU(N): "))
    su_max = int(input("Enter the maximum N of SU(N): "))
    so_min = int(input("Enter the minimum N of SO(N): "))
    so_max = int(input("Enter the maximum N of SO(N): "))
    sp_min = int(input("Enter the minimum N of Sp(N): "))
    sp_max = int(input("Enter the maximum N of Sp(N): "))

    if su_min > su_max or so_min > so_max or sp_min > sp_max:
        print("Invalid input. Minimum value should be less than maximum value.")
        sys.exit(1)

    # SU seiberg duals
    for n_c in range(su_min, su_max + 1):
        nf_lower, nf_upper = get_seiberg_conformal_window(LieGroup.SU, n_c)
        for nf in range(math.floor(nf_lower) + 1, math.ceil(nf_upper)):
            dual_n_c = nf - n_c
            if dual_n_c <= 1 or dual_n_c > su_max: # no dual
                continue



            t1 = f"SU{n_c}nf{nf}"
            t2 = f"SU{dual_n_c}nf{nf}"
            if (t1, t2) in used_pair or (t2, t1) in used_pair:
                continue

            seiberg.append((t1, t2))
            used_pair.add((t1, t2))

            print(f"Found Seiberg dual pair: {t1} <-> {t2}")

    used_pair.clear()

    # SO seiberg duals
    for n_c in range(so_min, so_max + 1):
        nf_lower, nf_upper = get_seiberg_conformal_window(LieGroup.SO, n_c)
        for nf in range(math.floor(nf_lower) + 1, math.ceil(nf_upper)):
            dual_n_c = nf - n_c + 4
            if dual_n_c <= 1 or dual_n_c > so_max: # no dual
                continue

            t1 = f"SO{n_c}nf{nf}"
            t2 = f"SO{dual_n_c}nf{nf}"
            if (t1, t2) in used_pair or (t2, t1) in used_pair:
                continue

            seiberg.append((t1, t2))
            used_pair.add((t1, t2))

            print(f"Found Seiberg dual pair: {t1} <-> {t2}")

    used_pair.clear()

    # Sp seiberg duals
    for n_c in range(sp_min, sp_max + 1):
        nf_lower, nf_upper = get_seiberg_conformal_window(LieGroup.Sp, n_c)
        for nf in range(math.floor(nf_lower) + 1, math.ceil(nf_upper)):
            dual_n_c = nf - n_c - 2
            if dual_n_c <= 0 or dual_n_c > sp_max: # no dual
                continue

            t1 = f"Sp{n_c}nf{nf}"
            t2 = f"Sp{dual_n_c}nf{nf}"
            if dual_n_c == 1: # Sp(1) is SU(2)
                t2 = f"SU2nf{nf}"
            if (t1, t2) in used_pair or (t2, t1) in used_pair:
                continue

            seiberg.append((t1, t2))
            used_pair.add((t1, t2))

            print(f"Found Seiberg dual pair: {t1} <-> {t2}")


    kutasov = [
        ("SU3adj1nf2", "SU5adj1nf2"),
        ("SU3adj1nf3", "SU3adj1nf3"),
        ("SU3adj1nf3", "SU6adj1nf3"),
        ("SU3adj1nf4", "SU5adj1nf4"),
        ("SU3adj1nf5", "SU7adj1nf5"),
        ("SU4adj1nf2", "SU2adj1nf2"),
        ("SU4adj1nf3", "SU5adj1nf3"),
        ("SU4adj1nf3", "SU8adj1nf3"),
        ("SU4adj1nf4", "SU4adj1nf4"),
        ("SU4adj1nf4", "SU8adj1nf4"),
        ("SU4adj1nf5", "SU6adj1nf5")
    ]

    with open(filename, "w") as f:
        f.write("Seiberg\n")
        for t1, t2 in seiberg:
            f.write(f"{t1}, {t2}\n")
        f.write("\n")

        f.write("Kutasov\n")
        for t1, t2 in kutasov:
            f.write(f"{t1}, {t2}\n")
