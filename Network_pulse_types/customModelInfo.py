
# =============================================================================
# This code contains pulse names, abbreviations and color codings. This module 
# can be imported to facilitate standardization of terms and plots throughout
# different codes.
# Rienk Zorgdrager, University of Twente, January 2024.
# =============================================================================

model_info = {
    'pulseSingle_Reference_OneCycle': {
        'abbreviation': 'Ref',
        'color': '#000000ff',
        'linestyle': 'solid',
    },
    'pulseSingle_Short_LowF': {
        'abbreviation': 'S1.7',
        'color': '#0072bdff',
        'linestyle': 'solid',
    },
    'pulseSingle_Short_MedF': {
        'abbreviation': 'S2.5',
        'color': '#0072bdff',
        'linestyle': 'dashed',
    },
    'pulseSingle_Short_HighF': {
        'abbreviation': 'S3.4',
        'color': '#0072bdff',
        'linestyle': 'dotted',
    },
    'pulseSingle_Long_LowF': {
        'abbreviation': 'L1.7',
        'color': '#000079ff',
        'linestyle': 'solid',
    },
    'pulseSingle_Long_MedF': {
        'abbreviation': 'L2.5',
        'color': '#000079ff',
        'linestyle': 'dashed',
    },
    'pulseSingle_Long_HighF': {
        'abbreviation': 'L3.4',
        'color': '#000079ff',
        'linestyle': 'dotted',
    },
    'pulseChirp_Short_Downsweep': {
        'abbreviation': 'SDC',
        'color': '#ff5252ff',
        'linestyle': 'dotted',
    },
    'pulseChirp_Short_Upsweep': {
        'abbreviation': 'SUC',
        'color': '#ff5252ff',
        'linestyle': 'dashed',
    },
    'pulseChirp_Long_Downsweep': {
        'abbreviation': 'LDC',
        'color': '#b00000ff',
        'linestyle': 'dotted',
    },
    'pulseChirp_Long_Upsweep': {
        'abbreviation': 'LUC',
        'color': '#b00000ff',
        'linestyle': 'dashed',
    },
    'pulseTrain_Short_Delay': {
        'abbreviation': 'DPT',
        'color': '#666666ff',
        'linestyle': 'solid',
    },
    'pulseChirp_Short_Downsweep_compressed': {
        'abbreviation': 'SDCc',
        'color': '#ff5252ff',
        #'color': '#ff8080ff',
        'linestyle': (0, (3, 1, 1, 1, 1, 1)), # densely dashdotdotted
    },
    'pulseChirp_Short_Upsweep_compressed': {
        'abbreviation': 'SUCc',
        'color': '#ff5252ff',
        #'color': '#ff3535ff',
        'linestyle': (0, (3, 1, 3, 1, 3, 1)), 
    },
    'pulseChirp_Long_Downsweep_compressed': {
        'abbreviation': 'LDCc',
        'color': '#b00000ff',
        #'color': '#e70000ff',
        'linestyle': (0, (3, 1, 1, 1, 1, 1)), # densely dashdotdotted
    },
    'pulseChirp_Long_Upsweep_compressed': {
        'abbreviation': 'LUCc',
        'color': '#b00000ff',
        #'color': '#9b0000ff',
        'linestyle': (0, (3, 1, 3, 1, 3, 1)), 
    },
        'pulseChirp_Short_Downsweep_linear': {
        'abbreviation': 'SDCL',
        'color': '#ff5252ff',
        'linestyle': 'dashdot',
    },
        'pulseSingle_Reference_OneCycle_linear': {
        'abbreviation': 'RefL',
        'color': '#000000ff',
        'linestyle': 'dashdot',
    },
        'pulseExpVal_Short_chirp4': {
        'abbreviation': 'exp_chirp',
        'color': '#ff5252ff',
        'linestyle': 'dashdotdotted',
    },
        'pulseExpVal_Short_SIP4': {
        'abbreviation': 'exp_SIP',
        'color': '#000000ff',
        'linestyle': 'dashdot',
    },
}

