use 5.10.0;
use strict;
use warnings;
# use DDP;

# for my $l_wc (100) {
# for my $l_wc (1000, 100, 10, 1, 0.1) {
# for my $dt_l (0.001) {
# for my $dt_l (0.1, 0.01, 0.001, 0.0001) {
open(my $fh, '>', 'config.py') or die $!;

my @data = (
    "# ---------------------------------------------------------------------------------------------------------------------",
    "# PyQuantum.Constants",
    "import PyQuantum.Constants as Constants",
    "# ---------------------------------------------------------------------------------------------------------------------",
    '',
    "",
    "capacity = 2",
    "",
    "n_atoms = 2",
    "",
    "wc = Constants.wc",
    "wa = Constants.wc",
    "",
    # "g_0 = 0.01",
    # "g_step = 0.01",
    # "g_1 = 1.0",
    "g_0 = 0.51",
    "g_step = 0.01",
    "g_1 = 1.0",

    # g_0 = 1
    # g_0 = g_step
    # "g_1 = 1",
    # "g_0 = g_1",
    "",
    "l_0 = 0.01",
    "l_1 = 1.0",
    "l_step = 0.01",

    # "l_0 = 0.001",
    # "l_1 = 0.01",
    # "l_step = 0.001",

    # "l = Constants.wc / $l_wc",
    # l = Constants.wc / 10000
    "",
    "dt = 0.01 * Constants.ns",
    # "dt = $dt_l / l",
    # dt = 0.01 / l
    "",
    "sink_limit = 0.95",
    "",
    "sink_precision = 1e-4",
    # "ro_trace_precision = 1e-4",
    "ampl_precision = 1e-3",

    "path = 'out/mix'",
);

print $fh (join("\n", @data));

close($fh);

# system("~/software/Python/bin/python3.7 mix_config.py");
# system("python3.7 mix_config.py");
# }
# }
# print `python3 -c 'print(123)'`;
# `python3 mix_config.py > 2`;
# qx('python3 mix_config.py');
