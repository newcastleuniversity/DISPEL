"""Test cases for :mod:`dispel.providers.bdh.tasks.draw`."""

import pytest

from dispel.data.flags import Flag
from dispel.providers.generic.tasks.draw import process_draw
from dispel.providers.generic.tasks.draw.shapes import SHAPE_SIZE
from tests.processing.helper import assert_level_values
from tests.providers.ads.tasks.test_draw import INVALID_FIXTURE


@pytest.mark.parametrize(
    "level, flags",
    [
        ("square_counter_clock-right", [INVALID_FIXTURE]),
        ("square_counter_clock-right-02", [INVALID_FIXTURE]),
        ("square_clock-right", []),
        ("square_clock-right-02", [INVALID_FIXTURE]),
        ("infinity-right", [INVALID_FIXTURE]),
        ("infinity-right-02", []),
        ("spiral-right", []),
        ("spiral-right-02", [INVALID_FIXTURE]),
        ("square_counter_clock-left", []),
        ("square_counter_clock-left-02", [INVALID_FIXTURE]),
        ("square_clock-left", [INVALID_FIXTURE]),
        ("square_clock-left-02", [INVALID_FIXTURE]),
        ("infinity-left", []),
        ("infinity-left-02", []),
        ("spiral-left", []),
        ("spiral-left-02", [INVALID_FIXTURE]),
    ],
)
def test_bdh_draw_orientation_flag(
    example_reading_processed_draw_orientation_invalid, level, flags
):
    """Ensure phone orientation is controlled with io format."""
    invs = example_reading_processed_draw_orientation_invalid.get_level(
        level
    ).get_flags()
    expected = [inv.format(level=level, percentage=0.0) for inv in flags]
    assert invs == expected


NON_CONTINUOUS = Flag(
    id_="draw-behavioral-deviation-continuous_drawing",
    reason="The drawing is not continuous, the user has lifted the finger"
    "during level {level}.",
    stop_processing=False,
)


@pytest.mark.parametrize(
    "level, flags",
    [
        ("square_counter_clock-right", [NON_CONTINUOUS]),
        ("square_counter_clock-right-02", [NON_CONTINUOUS]),
        ("square_clock-right", [NON_CONTINUOUS]),
        ("infinity-right", [NON_CONTINUOUS]),
        ("infinity-right-02", [NON_CONTINUOUS]),
        ("square_counter_clock-left", [NON_CONTINUOUS]),
        ("square_counter_clock-left-02", [NON_CONTINUOUS]),
        ("square_clock-left", [NON_CONTINUOUS]),
        ("square_clock-left-02", [NON_CONTINUOUS]),
        ("infinity-left", [NON_CONTINUOUS]),
        ("infinity-left-02", [NON_CONTINUOUS]),
        ("spiral-left", [NON_CONTINUOUS]),
        ("spiral-left-02", [NON_CONTINUOUS]),
    ],
)
def test_bdh_draw_non_continuous_flag(
    example_reading_processed_draw_non_continuous, level, flags
):
    """Ensure phone orientation is controlled with io format."""
    invs = example_reading_processed_draw_non_continuous.get_level(level).get_flags()
    expected = [inv.format(level=level) for inv in flags]
    assert invs == expected


@pytest.mark.parametrize(
    "level,expected",
    [
        (
            "infinity-left",
            {
                "draw-left_inf_first-dur_acc-mean": 3.235394738567922e-05,
                "draw-left_inf_first-dur_acc-median": 3.843209435001357e-05,
                "draw-left_inf_first-dur_acc_normed_combined": 1.4995744821723669,
                "draw-left_inf_first-sim-mean": 8.808244410083566,
                "draw-left_inf_first-sim-median": 7.415195060894362,
                "draw-left_inf_first-press-mean": 0.15703050792217255,
                "draw-left_inf_first-press-cv": 0.2645415961742401,
                "draw-left_inf_first-user_dur": 3509.0,
                "draw-left_inf_first-smooth": -2.3154744013967075,
                "draw-left_inf_first-rt": 3318.0,
                "draw-left_inf_first-speed-mean": 430.2757440832878,
                "draw-left_inf_first-speed-std": 122.74058477139023,
                "draw-left_inf_first-speed-median": 426.2237946192423,
                "draw-left_inf_first-speed-min": 0.0,
                "draw-left_inf_first-speed-max": 777.9981825086804,
                "draw-left_inf_first-speed-q95": 624.8202562332152,
                "draw-left_inf_first-speed-cv": 0.28526029286845306,
                "draw-left_inf_first-c_ratio": 1.0554436582801772,
                "draw-left_inf_first-raw_pacman_score": 0.572,
            },
        ),
        (
            "square_counter_clock-left",
            {
                "draw-left_scc_first-dur_acc-mean": 5.020474596886231e-05,
                "draw-left_scc_first-dur_acc-median": 5.530980669359665e-05,
                "draw-left_scc_first-dur_acc_normed_combined": 1.193019276541294,
                "draw-left_scc_first-sim-mean": 6.494436130436189,
                "draw-left_scc_first-sim-median": 5.895003718703971,
                "draw-left_scc_first-press-mean": 0.1285887062549591,
                "draw-left_scc_first-press-cv": 0.255656361579895,
                "draw-left_scc_first-user_dur": 3067.0,
                "draw-left_scc_first-smooth": -2.5229532601696105,
                "draw-left_scc_first-rt": 935.0,
                "draw-left_scc_first-speed-mean": 483.983228734315,
                "draw-left_scc_first-speed-std": 295.90439124206097,
                "draw-left_scc_first-speed-median": 490.35975337028503,
                "draw-left_scc_first-speed-min": 0.0,
                "draw-left_scc_first-speed-max": 1542.2271490097046,
                "draw-left_scc_first-speed-q95": 875.2379585714895,
                "draw-left_scc_first-speed-cv": 0.6113938948171677,
                "draw-left_scc_first-c_ratio": 0.9920476070703865,
                "draw-left_scc_first-itrem_ps_ene_diss": 23.250951697551322,
                "draw-left_scc_first-itrem_ps_ene_x_traj": 23.969657019604764,
                "draw-left_scc_first-itrem_ps_ene_y_traj": 495.04446849178527,
                "draw-left_scc_first-itrem_ps_peak_diss": 2.222311114666809,
                "draw-left_scc_first-itrem_ps_peak_x_traj": 2.222311114666809,
                "draw-left_scc_first-itrem_ps_peak_y_traj": 2.222311114666809,
                "draw-left_scc_first-itrem_ps_ent_diss": 1.3916037355420603,
                "draw-left_scc_first-itrem_ps_ent_x_traj": 1.4007766048829502,
                "draw-left_scc_first-itrem_ps_ent_y_traj": 0.7850912542048841,
                "draw-left_scc_first-itrem_ps_amp_diss": 14.749423701248423,
                "draw-left_scc_first-itrem_ps_amp_x_traj": 14.98919889816817,
                "draw-left_scc_first-itrem_ps_amp_y_traj": 418.58367220724466,
                "draw-left_scc_first-corner": -6.771230319383051,
                "draw-left_scc_first-axes_over": -1.4743702451082754,
                "draw-left_scc_first-raw_pacman_score": 0.7148362235067437,
            },
        ),
        (
            "square_clock-left",
            {
                "draw-left_sc_first-dur_acc-mean": 5.452548834320524e-05,
                "draw-left_sc_first-dur_acc-median": 6.58264556112442e-05,
                "draw-left_sc_first-dur_acc_normed_combined": 1.136132696940588,
                "draw-left_sc_first-sim-mean": 7.169682037977666,
                "draw-left_sc_first-sim-median": 5.938803946774588,
                "draw-left_sc_first-press-mean": 0.08288709819316864,
                "draw-left_sc_first-press-cv": 0.21158172190189362,
                "draw-left_sc_first-user_dur": 2558.0,
                "draw-left_sc_first-smooth": -1.7662182966685873,
                "draw-left_sc_first-rt": 736.0,
                "draw-left_sc_first-speed-mean": 583.913432479131,
                "draw-left_sc_first-speed-std": 320.6285591492473,
                "draw-left_sc_first-speed-median": 607.8437356387867,
                "draw-left_sc_first-speed-min": 0.0,
                "draw-left_sc_first-speed-max": 1291.666030883789,
                "draw-left_sc_first-speed-q95": 1065.000508626302,
                "draw-left_sc_first-speed-cv": 0.5491029000445311,
                "draw-left_sc_first-c_ratio": 0.9985133140473077,
                "draw-left_sc_first-itrem_ps_ene_diss": 15.712409363843395,
                "draw-left_sc_first-itrem_ps_ene_x_traj": 18.827599215519136,
                "draw-left_sc_first-itrem_ps_ene_y_traj": 513.9257161632304,
                "draw-left_sc_first-itrem_ps_peak_diss": 4.0001600064002565,
                "draw-left_sc_first-itrem_ps_peak_x_traj": 4.0001600064002565,
                "draw-left_sc_first-itrem_ps_peak_y_traj": 4.0001600064002565,
                "draw-left_sc_first-itrem_ps_ent_diss": 1.291609207298987,
                "draw-left_sc_first-itrem_ps_ent_x_traj": 0.9113521499203092,
                "draw-left_sc_first-itrem_ps_ent_y_traj": 0.8230034785969766,
                "draw-left_sc_first-itrem_ps_amp_diss": 7.8558904337344195,
                "draw-left_sc_first-itrem_ps_amp_x_traj": 9.413423055775256,
                "draw-left_sc_first-itrem_ps_amp_y_traj": 256.9525795672919,
                "draw-left_sc_first-corner": 0.40100693642848856,
                "draw-left_sc_first-axes_over": -2.359315430561471,
                "draw-left_sc_first-raw_pacman_score": 0.603082851637765,
            },
        ),
        (
            "square_clock-left-02",
            {
                "draw-left_sc_sec-dur_acc-mean": 6.6329461190158e-05,
                "draw-left_sc_sec-dur_acc-median": 7.90057140300197e-05,
                "draw-left_sc_sec-dur_acc_normed_combined": 1.0354918958225434,
                "draw-left_sc_sec-sim-mean": 6.40997319137424,
                "draw-left_sc_sec-sim-median": 5.381510353360765,
                "draw-left_sc_sec-press-mean": 0.07744717597961426,
                "draw-left_sc_sec-press-cv": 0.20493270456790924,
                "draw-left_sc_sec-user_dur": 2352.0,
                "draw-left_sc_sec-smooth": -1.7686459583239436,
                "draw-left_sc_sec-rt": 860.0,
                "draw-left_sc_sec-speed-mean": 645.5598624237598,
                "draw-left_sc_sec-speed-std": 355.67042446820886,
                "draw-left_sc_sec-speed-median": 649.727092069738,
                "draw-left_sc_sec-speed-min": 0.0,
                "draw-left_sc_sec-speed-max": 1416.666030883789,
                "draw-left_sc_sec-speed-q95": 1176.6339470358455,
                "draw-left_sc_sec-speed-cv": 0.5509487890601521,
                "draw-left_sc_sec-c_ratio": 1.0159585814606165,
                "draw-left_sc_sec-itrem_ps_ene_diss": 0.5390528244812464,
                "draw-left_sc_sec-itrem_ps_ene_x_traj": 0.5793196331892321,
                "draw-left_sc_sec-itrem_ps_ene_y_traj": 497.09889399396457,
                "draw-left_sc_sec-itrem_ps_peak_diss": 3.529552946823755,
                "draw-left_sc_sec-itrem_ps_peak_x_traj": 3.529552946823755,
                "draw-left_sc_sec-itrem_ps_peak_y_traj": 3.529552946823755,
                "draw-left_sc_sec-itrem_ps_ent_diss": 1.415224705428943,
                "draw-left_sc_sec-itrem_ps_ent_x_traj": 1.3198402009877617,
                "draw-left_sc_sec-itrem_ps_ent_y_traj": 0.9161822793384873,
                "draw-left_sc_sec-itrem_ps_amp_diss": 0.3054510486753514,
                "draw-left_sc_sec-itrem_ps_amp_x_traj": 0.32826799422887926,
                "draw-left_sc_sec-itrem_ps_amp_y_traj": 281.67810568831607,
                "draw-left_sc_sec-corner": 1.6473020528763584,
                "draw-left_sc_sec-axes_over": 0.4579200395444805,
                "draw-left_sc_sec-raw_pacman_score": 0.8410404624277457,
            },
        ),
        (
            "spiral-left",
            {
                "draw-left_spi_first-dur_acc-mean": 2.5103421784225442e-05,
                "draw-left_spi_first-dur_acc-median": 2.9994632810161447e-05,
                "draw-left_spi_first-dur_acc_normed_combined": 2.2433379014229438,
                "draw-left_spi_first-sim-mean": 15.530295077273813,
                "draw-left_spi_first-sim-median": 12.99777697649322,
                "draw-left_spi_first-press-mean": 0.0924837663769722,
                "draw-left_spi_first-press-cv": 0.3015757203102112,
                "draw-left_spi_first-user_dur": 2565.0,
                "draw-left_spi_first-smooth": -2.024722877952559,
                "draw-left_spi_first-rt": 2433.0,
                "draw-left_spi_first-speed-mean": 664.9347168340547,
                "draw-left_spi_first-speed-std": 290.17183252661397,
                "draw-left_spi_first-speed-median": 605.3081961239085,
                "draw-left_spi_first-speed-min": 0.0,
                "draw-left_spi_first-speed-max": 1510.380744934082,
                "draw-left_spi_first-speed-q95": 1229.9541049533425,
                "draw-left_spi_first-speed-cv": 0.4363914609665825,
                "draw-left_spi_first-c_ratio": 0.90742965248179,
                "draw-left_spi_first-cross": 4.0,
                "draw-left_spi_first-cross_per_sec": 1.574803149606299,
                "draw-left_spi_first-cross_freq-mean": 2.9492408577507896,
                "draw-left_spi_first-cross_freq-std": 2.727345446930613,
                "draw-left_spi_first-cross_freq-cv": 0.9247618551611269,
                "draw-left_spi_first-raw_pacman_score": 0.32605905006418484,
            },
        ),
        (
            "infinity-left-02",
            {
                "draw-left_inf_sec-dur_acc-mean": 4.457855144330559e-05,
                "draw-left_inf_sec-dur_acc-median": 4.912108551146925e-05,
                "draw-left_inf_sec-dur_acc_normed_combined": 1.7668706475498708,
                "draw-left_inf_sec-sim-mean": 13.250036959466378,
                "draw-left_inf_sec-sim-median": 12.024723152450646,
                "draw-left_inf_sec-press-mean": 0.09542080014944077,
                "draw-left_inf_sec-press-cv": 0.31446385383605957,
                "draw-left_inf_sec-user_dur": 1693.0,
                "draw-left_inf_sec-smooth": -2.3437419306002,
                "draw-left_inf_sec-rt": 839.0,
                "draw-left_inf_sec-speed-mean": 786.6046039575065,
                "draw-left_inf_sec-speed-std": 204.00446262632502,
                "draw-left_inf_sec-speed-median": 815.8550963682287,
                "draw-left_inf_sec-speed-min": 0.0,
                "draw-left_inf_sec-speed-max": 1133.3333333333333,
                "draw-left_inf_sec-speed-q95": 1023.143321275711,
                "draw-left_inf_sec-speed-cv": 0.2593481675545159,
                "draw-left_inf_sec-c_ratio": 0.9069488188443068,
                "draw-left_inf_sec-raw_pacman_score": 0.261,
            },
        ),
        (
            "square_counter_clock-left-02",
            {
                "draw-left_scc_sec-dur_acc-mean": 7.018062011351826e-05,
                "draw-left_scc_sec-dur_acc-median": 7.579923264537132e-05,
                "draw-left_scc_sec-dur_acc_normed_combined": 1.0388554837514115,
                "draw-left_scc_sec-sim-mean": 6.123312378977984,
                "draw-left_scc_sec-sim-median": 5.669422300302133,
                "draw-left_scc_sec-press-mean": 0.0955105721950531,
                "draw-left_scc_sec-press-cv": 0.28806188702583313,
                "draw-left_scc_sec-user_dur": 2327.0,
                "draw-left_scc_sec-smooth": -2.5251870557534994,
                "draw-left_scc_sec-rt": 675.0,
                "draw-left_scc_sec-speed-mean": 639.955221048022,
                "draw-left_scc_sec-speed-std": 410.7418146379205,
                "draw-left_scc_sec-speed-median": 595.1904720730251,
                "draw-left_scc_sec-speed-min": 0.0,
                "draw-left_scc_sec-speed-max": 1437.5,
                "draw-left_scc_sec-speed-q95": 1313.726088579963,
                "draw-left_scc_sec-speed-cv": 0.6418289922930382,
                "draw-left_scc_sec-c_ratio": 0.992379560951868,
                "draw-left_scc_sec-itrem_ps_ene_diss": 19.655019495749748,
                "draw-left_scc_sec-itrem_ps_ene_x_traj": 16.57083137126602,
                "draw-left_scc_sec-itrem_ps_ene_y_traj": 433.7244381182296,
                "draw-left_scc_sec-itrem_ps_peak_diss": 2.8572571474287543,
                "draw-left_scc_sec-itrem_ps_peak_x_traj": 2.8572571474287543,
                "draw-left_scc_sec-itrem_ps_peak_y_traj": 2.8572571474287543,
                "draw-left_scc_sec-itrem_ps_ent_diss": 1.0352288079531191,
                "draw-left_scc_sec-itrem_ps_ent_x_traj": 0.7352047686090777,
                "draw-left_scc_sec-itrem_ps_ent_y_traj": 0.8232240241921887,
                "draw-left_scc_sec-itrem_ps_amp_diss": 13.757963306478944,
                "draw-left_scc_sec-itrem_ps_amp_x_traj": 11.59911797660782,
                "draw-left_scc_sec-itrem_ps_amp_y_traj": 303.5949623984934,
                "draw-left_scc_sec-corner": -16.833712548095786,
                "draw-left_scc_sec-axes_over": 0.3267617921746506,
                "draw-left_scc_sec-raw_pacman_score": 0.7302504816955684,
            },
        ),
        (
            "spiral-left-02",
            {
                "draw-left_spi_sec-dur_acc-mean": 4.075232126969719e-05,
                "draw-left_spi_sec-dur_acc-median": 4.598883121749602e-05,
                "draw-left_spi_sec-dur_acc_normed_combined": 1.6527129666810865,
                "draw-left_spi_sec-sim-mean": 10.397660804524536,
                "draw-left_spi_sec-sim-median": 9.213733037818939,
                "draw-left_spi_sec-press-mean": 0.09700703620910645,
                "draw-left_spi_sec-press-cv": 0.25295817852020264,
                "draw-left_spi_sec-user_dur": 2360.0,
                "draw-left_spi_sec-smooth": -2.0436295171761385,
                "draw-left_spi_sec-rt": 841.0,
                "draw-left_spi_sec-speed-mean": 791.0492153788945,
                "draw-left_spi_sec-speed-std": 203.63583947809005,
                "draw-left_spi_sec-speed-median": 810.3610277175903,
                "draw-left_spi_sec-speed-min": 0.0,
                "draw-left_spi_sec-speed-max": 1185.8540773391724,
                "draw-left_spi_sec-speed-q95": 1043.540358543396,
                "draw-left_spi_sec-speed-cv": 0.257424993943712,
                "draw-left_spi_sec-c_ratio": 0.9872614844069052,
                "draw-left_spi_sec-cross": 10.0,
                "draw-left_spi_sec-cross_per_sec": 4.264392324093817,
                "draw-left_spi_sec-cross_freq-mean": 7.137463034689265,
                "draw-left_spi_sec-cross_freq-std": 8.447228210056705,
                "draw-left_spi_sec-cross_freq-cv": 1.183505703497414,
                "draw-left_spi_sec-raw_pacman_score": 0.4454428754813864,
            },
        ),
        ("square_counter_clock-right", {}),
        (
            "infinity-right",
            {
                "draw-right_inf_first-dur_acc-mean": 2.885272977471174e-05,
                "draw-right_inf_first-dur_acc-median": 3.718267296096485e-05,
                "draw-right_inf_first-dur_acc_normed_combined": 1.560078250447603,
                "draw-right_inf_first-sim-mean": 10.605497448902684,
                "draw-right_inf_first-sim-median": 8.229573821678352,
                "draw-right_inf_first-press-mean": 0.08921795338392258,
                "draw-right_inf_first-press-cv": 0.5465384721755981,
                "draw-right_inf_first-user_dur": 3268.0,
                "draw-right_inf_first-smooth": -2.3343298632227127,
                "draw-right_inf_first-rt": 790.0,
                "draw-right_inf_first-speed-mean": 476.0362100355548,
                "draw-right_inf_first-speed-std": 158.63357049284176,
                "draw-right_inf_first-speed-median": 481.22873405615485,
                "draw-right_inf_first-speed-min": 0.0,
                "draw-right_inf_first-speed-max": 987.7718687057495,
                "draw-right_inf_first-speed-q95": 736.5721434354781,
                "draw-right_inf_first-speed-cv": 0.33323845360627824,
                "draw-right_inf_first-c_ratio": 1.0688450799225266,
                "draw-right_inf_first-raw_pacman_score": 0.537,
            },
        ),
        (
            "infinity-right-02",
            {
                "draw-right_inf_sec-dur_acc-mean": 3.442792535524326e-05,
                "draw-right_inf_sec-dur_acc-median": 3.9275386454421306e-05,
                "draw-right_inf_sec-dur_acc_normed_combined": 1.5047700992951625,
                "draw-right_inf_sec-sim-mean": 8.863652188929597,
                "draw-right_inf_sec-sim-median": 7.76967926946915,
                "draw-right_inf_sec-press-mean": 0.12090562283992767,
                "draw-right_inf_sec-press-cv": 0.4650186002254486,
                "draw-right_inf_sec-user_dur": 3277.0,
                "draw-right_inf_sec-smooth": -2.358972920645066,
                "draw-right_inf_sec-rt": 1134.0,
                "draw-right_inf_sec-speed-mean": 444.18609826928787,
                "draw-right_inf_sec-speed-std": 171.05073900206696,
                "draw-right_inf_sec-speed-median": 442.92280077934265,
                "draw-right_inf_sec-speed-min": 20.833969116210938,
                "draw-right_inf_sec-speed-max": 924.9004476210649,
                "draw-right_inf_sec-speed-q95": 685.7850035031634,
                "draw-right_inf_sec-speed-cv": 0.3850880062850761,
                "draw-right_inf_sec-c_ratio": 1.0027572032452452,
                "draw-right_inf_sec-raw_pacman_score": 0.539,
            },
        ),
        (
            "square_clock-right",
            {
                "draw-right_sc_first-dur_acc-mean": 3.069139764525812e-05,
                "draw-right_sc_first-dur_acc-median": 3.053370648388196e-05,
                "draw-right_sc_first-dur_acc_normed_combined": 1.6758686639898075,
                "draw-right_sc_first-sim-mean": 8.858732932081558,
                "draw-right_sc_first-sim-median": 8.90448381021745,
                "draw-right_sc_first-press-mean": 0.14191175997257233,
                "draw-right_sc_first-press-cv": 0.4345552623271942,
                "draw-right_sc_first-user_dur": 3678.0,
                "draw-right_sc_first-smooth": -1.6905213346546892,
                "draw-right_sc_first-rt": 693.0,
                "draw-right_sc_first-speed-mean": 401.55206314453534,
                "draw-right_sc_first-speed-std": 192.023811692488,
                "draw-right_sc_first-speed-median": 428.1817238315258,
                "draw-right_sc_first-speed-min": 0.0,
                "draw-right_sc_first-speed-max": 1011.5902158949109,
                "draw-right_sc_first-speed-q95": 702.9510370072196,
                "draw-right_sc_first-speed-cv": 0.4782040221354077,
                "draw-right_sc_first-c_ratio": 0.9800880821777831,
                "draw-right_sc_first-itrem_ps_ene_diss": 0.0,
                "draw-right_sc_first-itrem_ps_ene_x_traj": 0.0,
                "draw-right_sc_first-itrem_ps_ene_y_traj": 0.0,
                "draw-right_sc_first-itrem_ps_peak_diss": 5.00020000800032,
                "draw-right_sc_first-itrem_ps_peak_x_traj": 5.00020000800032,
                "draw-right_sc_first-itrem_ps_peak_y_traj": 5.00020000800032,
                "draw-right_sc_first-itrem_ps_ent_diss": 1.915785394402599,
                "draw-right_sc_first-itrem_ps_ent_x_traj": 1.5688876781411343,
                "draw-right_sc_first-itrem_ps_ent_y_traj": 0.7573668130762277,
                "draw-right_sc_first-itrem_ps_amp_diss": 0.029752568338635734,
                "draw-right_sc_first-itrem_ps_amp_x_traj": 0.032736496713515334,
                "draw-right_sc_first-itrem_ps_amp_y_traj": 206.44678671458084,
                "draw-right_sc_first-corner": -12.568351790473445,
                "draw-right_sc_first-axes_over": 14.092041994291904,
                "draw-right_sc_first-raw_pacman_score": 0.46724470134874757,
            },
        ),
        (
            "square_clock-right-02",
            {
                "draw-right_sc_sec-dur_acc-mean": 3.491242071440589e-05,
                "draw-right_sc_sec-dur_acc-median": 4.716753706862887e-05,
                "draw-right_sc_sec-dur_acc_normed_combined": 1.315284193533544,
                "draw-right_sc_sec-sim-mean": 7.158985556047768,
                "draw-right_sc_sec-sim-median": 5.298930814586207,
                "draw-right_sc_sec-press-mean": 0.15386363863945007,
                "draw-right_sc_sec-press-cv": 0.4799802601337433,
                "draw-right_sc_sec-user_dur": 4001.0000000000005,
                "draw-right_sc_sec-smooth": -1.6784177383133736,
                "draw-right_sc_sec-rt": 693.0,
                "draw-right_sc_sec-speed-mean": 376.7975785683015,
                "draw-right_sc_sec-speed-std": 227.18180984049326,
                "draw-right_sc_sec-speed-median": 372.5496179917279,
                "draw-right_sc_sec-speed-min": 0.0,
                "draw-right_sc_sec-speed-max": 896.1877261891084,
                "draw-right_sc_sec-speed-q95": 725.4889993106617,
                "draw-right_sc_sec-speed-cv": 0.6029279983796721,
                "draw-right_sc_sec-c_ratio": 1.010181523707162,
                "draw-right_sc_sec-itrem_ps_ene_diss": 0.4790114360908459,
                "draw-right_sc_sec-itrem_ps_ene_x_traj": 0.32859920677724863,
                "draw-right_sc_sec-itrem_ps_ene_y_traj": 34.247551537314145,
                "draw-right_sc_sec-itrem_ps_peak_diss": 2.222311114666809,
                "draw-right_sc_sec-itrem_ps_peak_x_traj": 2.222311114666809,
                "draw-right_sc_sec-itrem_ps_peak_y_traj": 2.222311114666809,
                "draw-right_sc_sec-itrem_ps_ent_diss": 1.553361531751913,
                "draw-right_sc_sec-itrem_ps_ent_x_traj": 1.590641000135947,
                "draw-right_sc_sec-itrem_ps_ent_y_traj": 1.0739819627347178,
                "draw-right_sc_sec-itrem_ps_amp_diss": 0.37813827199656735,
                "draw-right_sc_sec-itrem_ps_amp_x_traj": 0.18449590427290263,
                "draw-right_sc_sec-itrem_ps_amp_y_traj": 29.824624980941024,
                "draw-right_sc_sec-corner": 0.1496892216023268,
                "draw-right_sc_sec-axes_over": 7.460016227846187,
                "draw-right_sc_sec-raw_pacman_score": 0.6955684007707129,
            },
        ),
        (
            "spiral-right",
            {
                "draw-right_spi_first-dur_acc-mean": 2.130408757944396e-05,
                "draw-right_spi_first-dur_acc-median": 2.5315945912340535e-05,
                "draw-right_spi_first-dur_acc_normed_combined": 2.028558496549988,
                "draw-right_spi_first-sim-mean": 12.710357136548172,
                "draw-right_spi_first-sim-median": 10.696126565471797,
                "draw-right_spi_first-press-mean": 0.09782806783914566,
                "draw-right_spi_first-press-cv": 0.33565208315849304,
                "draw-right_spi_first-user_dur": 3693.0,
                "draw-right_spi_first-smooth": -2.0889762968301,
                "draw-right_spi_first-rt": 2059.0,
                "draw-right_spi_first-speed-mean": 501.72694084340696,
                "draw-right_spi_first-speed-std": 219.94798526688479,
                "draw-right_spi_first-speed-median": 495.6234614638721,
                "draw-right_spi_first-speed-min": 0.0,
                "draw-right_spi_first-speed-max": 1060.6601238250732,
                "draw-right_spi_first-speed-q95": 873.8746523857112,
                "draw-right_spi_first-speed-cv": 0.4383818514850936,
                "draw-right_spi_first-c_ratio": 0.9811837378486042,
                "draw-right_spi_first-cross": 8.0,
                "draw-right_spi_first-cross_per_sec": 2.2573363431151243,
                "draw-right_spi_first-cross_freq-mean": 6.29647575847015,
                "draw-right_spi_first-cross_freq-std": 7.219504010557381,
                "draw-right_spi_first-cross_freq-cv": 1.1465944263893266,
                "draw-right_spi_first-raw_pacman_score": 0.38254172015404364,
            },
        ),
        (
            "spiral-right-02",
            {
                "draw-right_spi_sec-dur_acc-mean": 2.6111396676448492e-05,
                "draw-right_spi_sec-dur_acc-median": 3.283943339520017e-05,
                "draw-right_spi_sec-dur_acc_normed_combined": 1.5039583673468953,
                "draw-right_spi_sec-sim-mean": 7.975313098419043,
                "draw-right_spi_sec-sim-median": 6.341356789734764,
                "draw-right_spi_sec-press-mean": 0.12739619612693787,
                "draw-right_spi_sec-press-cv": 0.339141309261322,
                "draw-right_spi_sec-user_dur": 4802.0,
                "draw-right_spi_sec-smooth": -2.05459945453429,
                "draw-right_spi_sec-rt": 779.0,
                "draw-right_spi_sec-speed-mean": 385.9628698089922,
                "draw-right_spi_sec-speed-std": 133.06523307632068,
                "draw-right_spi_sec-speed-median": 369.96086905984316,
                "draw-right_spi_sec-speed-min": 0.0,
                "draw-right_spi_sec-speed-max": 931.6945672035217,
                "draw-right_spi_sec-speed-q95": 620.2880782239575,
                "draw-right_spi_sec-speed-cv": 0.34476174649176194,
                "draw-right_spi_sec-c_ratio": 0.9858126061018837,
                "draw-right_spi_sec-cross": 9.0,
                "draw-right_spi_sec-cross_per_sec": 1.8903591682419658,
                "draw-right_spi_sec-cross_freq-mean": 2.9926942304761655,
                "draw-right_spi_sec-cross_freq-std": 2.5905745701220684,
                "draw-right_spi_sec-cross_freq-cv": 0.865632894848026,
                "draw-right_spi_sec-raw_pacman_score": 0.576379974326059,
            },
        ),
        (
            "square_counter_clock-right-02",
            {
                "draw-right_scc_sec-dur_acc-mean": 5.3256641893539427e-05,
                "draw-right_scc_sec-dur_acc-median": 6.56053033349614e-05,
                "draw-right_scc_sec-dur_acc_normed_combined": 1.0996425600212976,
                "draw-right_scc_sec-sim-mean": 6.885588800987686,
                "draw-right_scc_sec-sim-median": 5.5895380153657275,
                "draw-right_scc_sec-press-mean": 0.1064242497086525,
                "draw-right_scc_sec-press-cv": 0.5598417520523071,
                "draw-right_scc_sec-user_dur": 2727.0,
                "draw-right_scc_sec-smooth": -2.537207974765995,
                "draw-right_scc_sec-rt": 781.0,
                "draw-right_scc_sec-speed-mean": 525.6584950061206,
                "draw-right_scc_sec-speed-std": 253.78020911033087,
                "draw-right_scc_sec-speed-median": 552.9868680667254,
                "draw-right_scc_sec-speed-min": 31.425768136978146,
                "draw-right_scc_sec-speed-max": 1013.7933492660522,
                "draw-right_scc_sec-speed-q95": 870.1483845710752,
                "draw-right_scc_sec-speed-cv": 0.48278532834778204,
                "draw-right_scc_sec-c_ratio": 0.9584286005641778,
                "draw-right_scc_sec-itrem_ps_ene_diss": 0.0,
                "draw-right_scc_sec-itrem_ps_ene_x_traj": 0.0,
                "draw-right_scc_sec-itrem_ps_ene_y_traj": 0.0,
                "draw-right_scc_sec-itrem_ps_peak_diss": 5.454763645091258,
                "draw-right_scc_sec-itrem_ps_peak_x_traj": 5.454763645091258,
                "draw-right_scc_sec-itrem_ps_peak_y_traj": 5.454763645091258,
                "draw-right_scc_sec-itrem_ps_ent_diss": 0.8915032332672334,
                "draw-right_scc_sec-itrem_ps_ent_x_traj": 0.8879305421731765,
                "draw-right_scc_sec-itrem_ps_ent_y_traj": 0.7621279179378665,
                "draw-right_scc_sec-itrem_ps_amp_diss": 7.416928092259604,
                "draw-right_scc_sec-itrem_ps_amp_x_traj": 7.425688588044996,
                "draw-right_scc_sec-itrem_ps_amp_y_traj": 137.2890328368033,
                "draw-right_scc_sec-corner": -18.437537992797086,
                "draw-right_scc_sec-axes_over": 0.3198489691265394,
                "draw-right_scc_sec-raw_pacman_score": 0.5895953757225434,
            },
        ),
        (
            "left-all",
            {
                "draw-left-acc_sig_ene": 0.9822224974632263,
                "draw-left-acc_ps_ene_x": 0.00039394264220948827,
                "draw-left-acc_ps_ene_y": 0.0004294070914578896,
                "draw-left-acc_ps_ene_z": 0.0009344483365936185,
                "draw-left-acc_ps_ene_xyz": 0.00045571915201048085,
                "draw-left-acc_ps_peak_x": 1.953125,
                "draw-left-acc_ps_peak_y": 2.109375,
                "draw-left-acc_ps_peak_z": 2.109375,
                "draw-left-acc_ps_peak_xyz": 0.078125,
                "draw-left-acc_ps_ent_x": 6.347075075397788,
                "draw-left-acc_ps_ent_y": 6.243908871064344,
                "draw-left-acc_ps_ent_z": 6.682002138249556,
                "draw-left-acc_ps_ent_xyz": 5.502976682805831,
                "draw-left-acc_ps_amp_x": 4.845608054893091e-05,
                "draw-left-acc_ps_amp_y": 4.240926864440553e-05,
                "draw-left-acc_ps_amp_z": 4.008360701845959e-05,
                "draw-left-acc_ps_amp_xyz": 0.0002385169646004215,
                "draw-left-gyr_sig_ene": 6.853268146514893,
                "draw-left-gyr_ps_ene_x": 0.023839214559018274,
                "draw-left-gyr_ps_ene_y": 0.011407959370473009,
                "draw-left-gyr_ps_ene_z": 0.00488718795535803,
                "draw-left-gyr_ps_ene_xyz": 0.015717935967103358,
                "draw-left-gyr_ps_peak_x": 0.546875,
                "draw-left-gyr_ps_peak_y": 0.390625,
                "draw-left-gyr_ps_peak_z": 0.390625,
                "draw-left-gyr_ps_peak_xyz": 0.15625,
                "draw-left-gyr_ps_ent_x": 5.630835313910378,
                "draw-left-gyr_ps_ent_y": 5.956811367946992,
                "draw-left-gyr_ps_ent_z": 5.44729416263925,
                "draw-left-gyr_ps_ent_xyz": 4.976426199935955,
                "draw-left-gyr_ps_amp_x": 0.0055802958086133,
                "draw-left-gyr_ps_amp_y": 0.003006913233548403,
                "draw-left-gyr_ps_amp_z": 0.0020267872605472803,
                "draw-left-gyr_ps_amp_xyz": 0.014196421019732952,
            },
        ),
        (
            "right-all",
            {
                "draw-right-acc_sig_ene": 0.6134737133979797,
                "draw-right-acc_ps_ene_x": 0.00019053449501371666,
                "draw-right-acc_ps_ene_y": 0.0001506397477235577,
                "draw-right-acc_ps_ene_z": 0.0004459558263114616,
                "draw-right-acc_ps_ene_xyz": 0.00014731901101905365,
                "draw-right-acc_ps_peak_x": 2.109375,
                "draw-right-acc_ps_peak_y": 2.109375,
                "draw-right-acc_ps_peak_z": 8.90625,
                "draw-right-acc_ps_peak_xyz": 0.3125,
                "draw-right-acc_ps_ent_x": 6.578206248296675,
                "draw-right-acc_ps_ent_y": 6.4959153323970344,
                "draw-right-acc_ps_ent_z": 6.7602629035158515,
                "draw-right-acc_ps_ent_xyz": 6.419417264778345,
                "draw-right-acc_ps_amp_x": 1.0500065400265157e-05,
                "draw-right-acc_ps_amp_y": 9.70161272562109e-06,
                "draw-right-acc_ps_amp_z": 1.748384784150403e-05,
                "draw-right-acc_ps_amp_xyz": 1.5469826394109987e-05,
                "draw-right-gyr_sig_ene": 4.198453426361084,
                "draw-right-gyr_ps_ene_x": 0.006074368940289787,
                "draw-right-gyr_ps_ene_y": 0.00648962693801991,
                "draw-right-gyr_ps_ene_z": 0.0026822038172014118,
                "draw-right-gyr_ps_ene_xyz": 0.004656322717888628,
                "draw-right-gyr_ps_peak_x": 0.390625,
                "draw-right-gyr_ps_peak_y": 0.78125,
                "draw-right-gyr_ps_peak_z": 0.390625,
                "draw-right-gyr_ps_peak_xyz": 0.15625,
                "draw-right-gyr_ps_ent_x": 6.597810182720707,
                "draw-right-gyr_ps_ent_y": 6.0207932245758204,
                "draw-right-gyr_ps_ent_z": 5.626188775347248,
                "draw-right-gyr_ps_ent_xyz": 5.334435920405051,
                "draw-right-gyr_ps_amp_x": 0.0004121391102671623,
                "draw-right-gyr_ps_amp_y": 0.0014505163999274373,
                "draw-right-gyr_ps_amp_z": 0.000792353879660368,
                "draw-right-gyr_ps_amp_xyz": 0.0027474164962768555,
            },
        ),
        (
            "all_levels",
            {
                "draw-press-mean": 0.1150931566953659,
                "draw-press-std": 0.05177358165383339,
                "draw-press-median": 0.10499999672174454,
                "draw-press-min": 0.0,
                "draw-press-max": 0.3149999976158142,
                "draw-press-cv": 0.4498406648635864,
                "draw-rt-mean": 1171.0666666666666,
                "draw-rt-std": 788.6172769625388,
                "draw-rt-median": 839.0,
                "draw-rt-min": 675.0,
                "draw-rt-max": 3318.0,
            },
        ),
    ],
)
def test_draw_process_bdh(example_reading_processed_draw, level, expected):
    """Unit test to ensure the drawing features are well computed."""
    assert_level_values(example_reading_processed_draw, level, expected)


@pytest.mark.xfail
def test_draw_process_bdh_bug(example_reading_processed_draw_bug):
    """Test drawing record with empty accelerometer data."""
    _ = example_reading_processed_draw_bug


def test_draw_process_bdh_no_corner(example_reading_processed_draw_bug_no_corner):
    """Test drawing record with no corner detected."""
    _ = example_reading_processed_draw_bug_no_corner


def test_draw_process_bdh_features(example_reading_processed_draw_bug_parsing):
    """Test drawing record with existing features in the json file."""
    _ = example_reading_processed_draw_bug_parsing


def test_draw_process_one_hand(example_reading_draw_one_hand):
    """Test processing a drawing record with only one hand."""
    _ = process_draw(example_reading_draw_one_hand)


def test_draw_opp_direction_and_excessive_overshoot(
    example_reading_processed_draw_opposite_direction,
):
    """Test processing a drawing with opposite direction."""
    _ = process_draw(example_reading_processed_draw_opposite_direction)
    example_reading_processed_draw_opposite_direction.get_level(
        "infinity-right-02"
    ).get_flags("draw-behavioral-deviation-opp_direction")
    example_reading_processed_draw_opposite_direction.get_level(
        "spiral-right"
    ).get_flags("draw-technical-deviation-val_user_path")
    example_reading_processed_draw_opposite_direction.get_level(
        "spiral-right-02"
    ).get_flags("draw-technical-deviation-excessive_overshoot_removal")


def test_reference_path_size(example_reading_processed_draw):
    """Test the good computation of reference path size."""
    for level_id in [
        "infinity-right",
        "spiral-right",
        "square_clock-right",
        "square_counter_clock-right",
    ]:
        level = example_reading_processed_draw.get_level(level_id)
        shape = level_id.split("-", maxsplit=1)[0]
        ref = level.get_raw_data_set("shape").data.iloc[0].values[0].reference
        assert len(ref) == SHAPE_SIZE["ADS"][shape]