import pstats

p = pstats.Stats('profile_output.prof')
p.sort_stats('cumulative').print_stats(10)