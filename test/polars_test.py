from common.balanced_sample_tool import TheorySampler


sampler = TheorySampler('../landscape_all.csv')
print(sampler.a_range())
print(sampler.c_range())
sampler.draw_central_charge_histogram(0.1)
print(sampler.get_theory_stats())

balanced_sample = sampler.get_balanced_sample((0.5, 1.5), (0.5, 1.5), 50)
print(balanced_sample.get_theory_stats())

manual_sample = sampler.get_manual_sample([
    "SU5s1S1a1A1nf1",
    "H0H1",
    "SU2nf4",
    "SU6s1S1a1A1nf0",
    "Sp2adj1a2nf4"
], 50)
print(manual_sample.get_theory_stats())