
# =============================================================================
# This code contains pulse names, abbreviations and color codings. This module 
# can be imported to facilitate standardization of terms and plots throughout
# different codes.
# Rienk Zorgdrager, University of Twente, January 2024.
# =============================================================================

model_info = {
    'pulseSingle_Reference_OneCycle': {
        'abbreviation': 'Ref',
        'color': '#45b41fff',
        'linestyle': 'solid',
    },
    'pulseSingle_Short_LowF': {
        'abbreviation': 'S1.7',
        'color': '#6ce7ffff',
        'linestyle': 'solid',
    },
    'pulseSingle_Short_MedF': {
        'abbreviation': 'S2.5',
        'color': '#1adaffff',
        'linestyle': 'dashed',
    },
    'pulseSingle_Short_HighF': {
        'abbreviation': 'S3.4',
        'color': '#00bfe4ff',
        'linestyle': 'dotted',
    },
    'pulseSingle_Long_LowF': {
        'abbreviation': 'L1.7',
        'color': '#6483ffff',
        'linestyle': 'solid',
    },
    'pulseSingle_Long_MedF': {
        'abbreviation': 'L2.5',
        'color': '#335dffff',
        'linestyle': 'dashed',
    },
    'pulseSingle_Long_HighF': {
        'abbreviation': 'L3.4',
        'color': '#0035ffff',
        'linestyle': 'dotted',
    },
    'pulseChirp_Short_Downsweep': {
        'abbreviation': 'SDC',
        'color': '#ff8080ff',
        'linestyle': 'dashdot',
    },
    'pulseChirp_Short_Upsweep': {
        'abbreviation': 'SUC',
        'color': '#ff3535ff',
        'linestyle': 'dashdot',
    },
    'pulseChirp_Long_Downsweep': {
        'abbreviation': 'LDC',
        'color': '#e70000ff',
        'linestyle': 'dashdot',
    },
    'pulseChirp_Long_Upsweep': {
        'abbreviation': 'LUC',
        'color': '#9b0000ff',
        'linestyle': 'dashdot',
    },
    'pulseTrain_Short_Delay': {
        'abbreviation': 'DPT',
        'color': '#666666ff',
        'linestyle': 'solid',
    },
    'pulseChirp_Short_Downsweep_compressed': {
        'abbreviation': 'SDCc',
        'color': '#eed74dff',
        #'color': '#ff8080ff',
        'linestyle': (0, (3, 1, 1, 1, 1, 1)), # densely dashdotdotted
    },
    'pulseChirp_Short_Upsweep_compressed': {
        'abbreviation': 'SUCc',
        'color': '#bca93cff',
        #'color': '#ff3535ff',
        'linestyle': (0, (3, 1, 1, 1, 1, 1)), # densely dashdotdotted
    },
    'pulseChirp_Long_Downsweep_compressed': {
        'abbreviation': 'LDCc',
        'color': '#998931ff',
        #'color': '#e70000ff',
        'linestyle': (0, (3, 1, 1, 1, 1, 1)), # densely dashdotdotted
    },
    'pulseChirp_Long_Upsweep_compressed': {
        'abbreviation': 'LUCc',
        'color': '#766a25ff',
        #'color': '#9b0000ff',
        'linestyle': (0, (3, 1, 1, 1, 1, 1)), # densely dashdotdotted
    },
        'pulseChirp_Short_Downsweep_linear': {
        'abbreviation': 'SDCL',
        'color': '#666666ff',
        'linestyle': 'dashdot',
    },
        'pulseSingle_Reference_OneCycle_linear': {
        'abbreviation': 'RefL',
        'color': '#45b41fff',
        'linestyle': 'dashdot',
    },
}
