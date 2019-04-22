"""Module to process dataframe

Contains functions to help study, visualize and understand datasets.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

def combine(room_name, value_adds):
    long_name = room_name
    if pd.notnull(value_adds):
        long_name = long_name + ' ' + value_adds
    tokens = word_tokenize(text_preprocess(long_name))
    return  " ".join(tokens)

def add_feature(feature_name, value_adds):
    for key, value in dict_labels.items():
        is_found = raw_data['combined'].str.contains(key)
        raw_data['rmf_no_smoking']['']=1
        raw_data['rmf_no_smoking'][~is_found]=0
        
def show_label(df_data, pattern):
    return df_data.loc[(df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined']))]

def add_label_by_label(df_data, labeled, pattern, label):
    df_data.loc[(df_data['labels'].str.contains(labeled)) & (df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])), 'labels'] = \
    df_data.loc[(df_data['labels'].str.contains(labeled)) & (df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])), 'labels'] + ' ' + label
    df_data.loc[(df_data['labels'].str.contains(labeled)) & (df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])) & (~pd.notnull(df_data['labels'])), 'labels'] = label

def add_label_if_not_present(df_data, labeled, pattern, label):
    df_data.loc[~(df_data['labels'].str.contains(labeled) & pd.notnull(df_data['labels'])) & df_data['combined'].str.contains(pattern) & pd.notnull(df_data['combined']), 'labels'] = \
    df_data.loc[~(df_data['labels'].str.contains(labeled) & pd.notnull(df_data['labels'])) & df_data['combined'].str.contains(pattern) & pd.notnull(df_data['combined']), 'labels'] + ' ' + label
    df_data.loc[(df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])) & (~pd.notnull(df_data['labels'])), 'labels'] = label
    
    
def add_label_by_column_name(df_data, name, pattern, label):
    df_data.loc[(df_data[name].str.contains(pattern)) & (pd.notnull(df_data[name])), 'labels'] = \
    df_data.loc[(df_data[name].str.contains(pattern)) & (pd.notnull(df_data[name])), 'labels'] + ' ' + label
    df_data.loc[(df_data[name].str.contains(pattern)) & (pd.notnull(df_data[name])) & (~pd.notnull(df_data['labels'])), 'labels'] = label

def add_label_if_labels_is_null(df_data, pattern, label):
    df_data.loc[(df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])) & (~pd.notnull(df_data['labels'])), 'labels'] = label
    
def add_label(df_data, pattern, label):
    df_data.loc[(df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])), 'labels'] = \
    df_data.loc[(df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])), 'labels'] + ' ' + label
    df_data.loc[(df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])) & (~pd.notnull(df_data['labels'])), 'labels'] = label
    
def clean_only_combined_with_label(df_data, label, pattern, replace_value=''):
    df_data.loc[(df_data['labels'].str.contains(label)) & (df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])),'combined'] = \
    df_data.loc[(df_data['labels'].str.contains(label)) & (df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])),'combined'].str.replace(pattern,replace_value)

def clean_only_combined(df_data, pattern, replace_value=''):
    df_data.loc[(df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])),'combined'] = \
    df_data.loc[(df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])),'combined'].str.replace(pattern,replace_value)

def clean_combined(df_data, pattern, label, replace_pattern=None, replace_value=''):
    if replace_pattern==None:
        replace_pattern = pattern
    df_data.loc[(df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])) & (df_data['labels'].str.contains(label)),'combined'] = \
    df_data.loc[(df_data['combined'].str.contains(pattern)) & (pd.notnull(df_data['combined'])) & (df_data['labels'].str.contains(label)),'combined'].str.replace(replace_pattern,replace_value)
    
def clean_only_combined_by_index(df_data, index, pattern, replace_value=''):
    df_data.loc[index,'combined'] = df_data.loc[index,'combined'].replace(pattern,replace_value)

def clean_column_by_column_name(df_data, column_name, pattern, replace_column, replace_pattern, replace_value=''):
    df_data.loc[(df_data[column_name].str.contains(pattern)) & (pd.notnull(df_data[column_name])) & (df_data[replace_column].str.contains(replace_pattern)),replace_column] = \
    df_data.loc[(df_data[column_name].str.contains(pattern)) & (pd.notnull(df_data[column_name])) & (df_data[replace_column].str.contains(replace_pattern)),replace_column].str.replace(replace_pattern,replace_value)

def clean_label_by_column_name(df_data, column_name, pattern, label, replace_value=''):
    df_data.loc[(df_data[column_name].str.contains(pattern)) & (pd.notnull(df_data[column_name])) & (df_data['labels'].str.contains(label)),'labels'] = \
    df_data.loc[(df_data[column_name].str.contains(pattern)) & (pd.notnull(df_data[column_name])) & (df_data['labels'].str.contains(label)),'labels'].str.replace(label,replace_value)

def clean_label_by_index(df_data, index, pattern, replace_value=''):
    df_data.loc[index, 'labels'] = df_data.loc[index, 'labels'].replace(pattern,replace_value)

def add_label_and_clean_by_index(df_data, index, label, pattern, replace_value=''):
    if pd.notnull(df_data.loc[index, 'labels']):
        df_data.loc[index, 'labels'] = df_data.loc[index, 'labels'] + ' ' + label
    else:
        df_data.loc[index, 'labels'] = label
    df_data.loc[index,'combined'] = df_data.loc[index,'combined'].replace(pattern,replace_value)

def add_label_and_clean_if_labels_is_null(df_data, pattern, label, replace_pattern=None, replace_value=''):
    add_label_if_labels_is_null(df_data, pattern, label)
    clean_only_combined_with_label(df_data, label, pattern, replace_value)
    
    
def add_label_and_clean(df_data, pattern, label, replace_pattern=None, replace_value=''):
    add_label(df_data, pattern, label)
    clean_combined(df_data, pattern, label, replace_pattern, replace_value)      
    
def add_label_and_clean_by_labeled(df_data, labeled, pattern, label, replace_value=''):
    add_label_by_label(df_data, labeled, pattern, label)
    clean_only_combined_with_label(df_data, label, pattern, replace_value)

def add_label_and_clean_if_label_not_present(df_data, labeled, pattern, label, replace_value=''):
    add_label_if_not_present(df_data, labeled, pattern, label)
    clean_only_combined_with_label(df_data, label, pattern, replace_value)
    

def remove_dup_labels(df_data):
    df_data.loc[pd.notnull(df_data['labels']), 'labels']=df_data.loc[pd.notnull(df_data['labels']), 'labels'].apply(remove_dups)    
    
    
def remove_dups(name):
    if len(name)<20:
        return name
    ulist = []
    [ulist.append(x) for x in name.replace(',', ' ').split() if x not in ulist]    
    return ' '.join(list(dict.fromkeys(ulist)))  

# Usage examples:

# raw_data.loc[(pd.notnull(raw_data['combined'])),'combined'] = raw_data.loc[(pd.notnull(raw_data['combined'])),'combined'].apply( lambda x: ' '.join(x.split()) )

# raw_data['labels'] = ''
# raw_data.head()
# raw_data[~pd.notnull(raw_data['combined'])]
# raw_data.loc[(raw_data['combined'].str.contains("accessible mobility & hear")) & (pd.notnull(raw_data['combined']))]
# raw_data.loc[(raw_data['combined'].str.contains('accessible roll shower|accessible bathtub|accessible shower|accessible ensuite|mobility accessible tub|accessible tub')) & (pd.notnull(raw_data['combined']))]
# raw_data.loc[(raw_data['combined'].str.contains('access')) & (pd.notnull(raw_data['combined']))]
# raw_data.loc[(raw_data['combined'].str.contains('breakfast')) & (pd.notnull(raw_data['combined']))]
# raw_data.loc[(raw_data['hotel_id']==54) & (pd.notnull(raw_data['combined']))]
# raw_data.loc[(raw_data['combined'].str.contains('mobility access.*|disable access')) & (pd.notnull(raw_data['combined']))]
# raw_data.loc[(raw_data['combined'].str.contains('mobility access')) & (pd.notnull(raw_data['combined']))]

# raw_data.loc[(raw_data['combined'].str.contains('mobility')) & (pd.notnull(raw_data['combined'])) & (raw_data['labels'].str.contains('rmf_accessible_mobility'))]

# raw_data.loc[(raw_data['combined'].str.contains('breakfast')) & (pd.notnull(raw_data['combined']))]

# raw_data.loc[(raw_data['combined'].str.contains('mobility (hr|hea) access')) & (pd.notnull(raw_data['combined']))]


# add_label(raw_data, 'bed access','rmf_accessible_mobility')
# clean_combined(raw_data, 'bed access','rmf_accessible_mobility','access','')
# raw_data.loc[(raw_data['combined'].str.contains('access')) & (pd.notnull(raw_data['combined']))]
# raw_data.loc[(raw_data['combined'].str.contains('1 king bed access')) & (pd.notnull(raw_data['combined']))]



# show_label(raw_data, 'bed access|bed accessories')
# raw_data.loc[(raw_data['combined'].str.contains('bed access')) & (pd.notnull(raw_data['combined']))]

# raw_data.loc[1200619]

# clean_only_combined(raw_data, 'access stairs|park lot access|steam bath|prkng lot access')
# raw_data.loc[(raw_data['combined'].str.contains('access')) & (pd.notnull(raw_data['combined']))]

# raw_data.loc[(raw_data['room_name'].str.contains('Two Queen')) & (raw_data['labels'].str.contains('bdc_3')) & (pd.notnull(raw_data['combined'])) & (pd.notnull(raw_data['labels'])), 'labels'] = \
# raw_data.loc[(raw_data['room_name'].str.contains('Two Queen')) & (raw_data['labels'].str.contains('bdc_3')) & (pd.notnull(raw_data['combined'])) & (pd.notnull(raw_data['labels'])), 'labels'].str.replace('bdc_3','bdc_2')

# add_label(raw_data, 'four queen bed','bdc_4')
# add_label_and_clean(raw_data, 'four queen bed','bdt_queen')
# add_label(raw_data, '4 queen bed','bdc_4')
# add_label_and_clean(raw_data, '4 queen bed','bdt_queen')
# add_label(raw_data, '4 queen','bdc_4')
# add_label_and_clean(raw_data, '4 queen','bdt_queen')
# add_label(raw_data, 'queen 1 single bed','bdc_1')
# add_label_and_clean(raw_data, 'queen 1 single bed','bdt_queen')
# clean_combined(raw_data, 'double queen','bdc_2 bdt_queen')
# clean_only_combined(raw_data, 'queen \\d+', 'queen')
# add_label_and_clean(raw_data, 'double queen size bed','bdc_2 bdt_queen')
# clean_only_combined(raw_data, 'queen size|queen bed', 'queen')
# add_label_and_clean(raw_data, 'share bathroom \\d+ queen','bdc_1 bdt_queen')
# add_label_and_clean(raw_data, 'city view','rmv_good')
# add_label_and_clean(raw_data, 'bedroom[s]? \\d+ queen','bdc_more bdt_queen',' \\d+ queen','')
# add_label_and_clean(raw_data, 'free wifi','rmf_free_internet')
# clean_only_combined(raw_data, 'rm 5 queen', 'queen')
# add_label_and_clean(raw_data, 'bedrooms 5 queen','bdc_more bdt_queen','bedrooms 5 queen', 'bedroom')
# add_label_and_clean(raw_data, 'villa 10 queen','bdc_more bdt_queen','villa 10 queen', 'villa')
# add_label_and_clean(raw_data, 'villa garden 5 queen','bdc_more bdt_queen','villa garden 5 queen', 'villa garden')
# add_label_and_clean(raw_data, 'vacation rental 5 queen','bdc_more bdt_queen','vacation rental 5 queen', 'vacation_rental')

# add_label_and_clean(raw_data, 'two bedroom','rmc_2 rmt_bedroom','two bedroom', '')
# add_label_and_clean(raw_data, 'one bedroom','rmc_1 rmt_bedroom','one bedroom', '')

# add_label_and_clean(raw_data, '1840 queen|30 queen|18 queen|19 queen','bdc_1 bdt_queen')
# add_label_and_clean(raw_data, ' sleep 6 queen','bdc_1 bdt_queen')
# add_label_and_clean(raw_data, '10 queen deluxe netflix','bdc_1 bdt_queen')

# add_label_and_clean_by_index(raw_data, 1311278, 'bdc_more bdt_queen', '8 queen single bed', '')
# add_label_and_clean_by_index(raw_data, 1319709, 'bdc_1 bdt_queen', '5 queen', '')
# add_label_and_clean_by_index(raw_data, 1351458, 'bdc_1 bdt_queen', '7 queen', '')
# raw_data.loc[306334,'combined'] = raw_data.loc[306334,'combined'].replace('8 queen','')

# clean_only_combined_by_index(raw_data, 1327480, '1a 1b', '')


# add_label_and_clean_by_index(raw_data, 1309539, 'bdc_more bdt_queen', '5 queen', '')
# add_label_and_clean_by_index(raw_data, 1309539, 'rmc_more rmt_bedroom', '6 bedrooms', '')
# add_label_and_clean_by_index(raw_data, 1327480, 'bdc_more bdt_queen rml_loft', 'family loft 5 queen', '')

# add_label_and_clean(raw_data, 'queen','bdc_1 bdt_queen')

# raw_data.loc[pd.notnull(raw_data['labels']), 'labels']=raw_data.loc[pd.notnull(raw_data['labels']), 'labels'].apply(remove_dups)

# add_label_and_clean(raw_data, '1 king bed|1 king|one king','bdc_1 bdt_king')
# add_label_and_clean(raw_data, '1 bedroom|one bedroom','rmc_1 rmt_bedroom')
# add_label_and_clean(raw_data, '2 king bed|2 king|two king','bdc_2 bdt_king')
# add_label_and_clean(raw_data, '2 bedroom|two bedroom','rmc_2 rmt_bedroom')
# add_label_and_clean(raw_data, 'double king bed|double king','bdc_2 bdt_king')
# add_label_and_clean(raw_data, '3 king bed|3 king|two king|triple king','bdc_3 bdt_king')
# add_label_and_clean(raw_data, '3 bedroom|three bedroom','rmc_3 rmt_bedroom')
# add_label_and_clean(raw_data, '4 king bed|4 king|four king|four king bed','bdc_4 bdt_king')
# add_label_and_clean(raw_data, '4 bedroom|four bedroom','rmc_4 rmt_bedroom')

# add_label_and_clean(raw_data, 'diamante view','rmv_better')
# add_label_and_clean(raw_data, 'resort view','rmv_good')

# add_label_and_clean(raw_data, 'penthouse 2 6 bedrooms|penthouse 6 bedrooms','rmc_more rmt_bedroom rmt_penthouse')
# add_label_and_clean(raw_data, 'semi inclusive','prmf_all_inclusive_semi')

# add_label_and_clean(raw_data, 'club 89 king size bed|sanctuary 55 king|sanctuary 90 king|icon 36 king|35 king|25 king|15 king bed|16 king bed','bdc_1 bdt_king')

# add_label_and_clean(raw_data, 'penn 5000 king','bdc_1 bdt_king')
# add_label_and_clean(raw_data, '5 king bed internet','bdc_more bdt_king rmt_penthouse rmf_free_internet')
# clean_only_combined(raw_data, 'sleep 6|sleep 8|grand cayman ', '')

# add_label_and_clean(raw_data, '5 bedrooms 5 king bed','rmc_more bdc_more bdt_king')
# add_label_and_clean(raw_data, 'six bedroom residence villa 5 kings','rmc_more bdc_more bdt_king rmf_terrace rmt_villa')
# add_label_and_clean(raw_data, 'terrace \\d+ king single bed','bdc_1 bdt_king rmf_terrace')
# add_label_and_clean(raw_data, 'beach view beachfront 5 king inclusive','bdc_more bdt_king rmv_better prmf_all_inclusive rml_direct_access')
# add_label_and_clean(raw_data, 'apartment','rmt_apartment')

# add_label_and_clean(raw_data, 'deluxe twin king april 2019 king bed','bdc_1 bdt_king rmt_delux')
# add_label_and_clean_by_index(raw_data, 1052573, 'bdc_more bdt_king rmf_terrace', '6 king twin terrace', '')
# add_label_and_clean_by_index(raw_data, 1052575, 'bdc_more bdt_king rmt_suite rmf_balcony', '7 king suite private balcony', '')
# add_label_and_clean_by_index(raw_data, 1214253, 'bdc_more bdt_king', '5 king', '')
# clean_only_combined_by_index(raw_data, 396024, 'private bathroom flat 5 b @ 128 kings rd', '')
# add_label_and_clean(raw_data, 'aloft king|king bed|king size bed|king', 'bdc_1 bdt_king')


# add_label_and_clean(raw_data, '1bdrm|1 bedroom|one bedroom','rmc_1 rmt_bedroom')
# add_label_and_clean(raw_data, 'sofabed|sofabd|extra bed','rmf_extra_bedding')

# add_label_and_clean(raw_data, 'spectacular view','rmv_best')
# add_label_and_clean(raw_data, 'partial sea view|partial ocean view','rmv_good')
# add_label_and_clean(raw_data, 'bay view waterfront|chesapeake bay view|sea view','rmv_better')
# add_label_and_clean(raw_data, 'lagoon view','rmv_good')
# add_label_and_clean(raw_data, 'garden view','rmv_standard')

# add_label_and_clean(raw_data, '1 double bed|1 double|one double','bdc_1 bdt_double')
# add_label_and_clean(raw_data, '2 double bed|2 double|two double|double double|dbl double','bdc_2 bdt_double')

# add_label_and_clean(raw_data, '3 double bed|3 double|three double|triple double','bdc_3 bdt_double')
# add_label_and_clean(raw_data, '4 double bed|4 double|four double|quadraple double','bdc_4 bdt_double')
# add_label_and_clean(raw_data, '2 bdrm|two bdrm','rmc_2 rmt_bedroom')

# add_label_and_clean(raw_data, '30 double bed|8 double open plan shower|725 double|club 89 double size bed|sanctuary 55 double|sanctuary 90 double|icon 36 double|35 double|25 double|15 double bed|16 double bed','bdc_1 bdt_double')
# add_label_and_clean(raw_data, 'club 89 double bed','bdc_1 bdt_double')
# add_label_and_clean(raw_data, 'comfort rm 7 double bed|comfort rm 8 double bed|comfort rm 6 double bed','bdc_1 bdt_double rmt_room')
# add_label_and_clean(raw_data, '8 double twin free welcome drink','bdc_1 bdt_double rmt_room prmf_welcome_item')
# add_label_and_clean(raw_data, '6 double bed dormitory mix','bdc_more bdt_double')
# add_label_and_clean(raw_data, 'villa s 8 double bed','bdc_more bdt_double rmt_villa')
# add_label_and_clean(raw_data, 'twin double april 2019 double bed','bdc_1 bdt_double')
# add_label_and_clean(raw_data, 'zs1 2 5 double petrinjska|6 double shower','bdc_1 bdt_double')

# add_label_and_clean_by_index(raw_data, 355853, 'bdc_1 bdt_double rmf_terrace', 'terrace 8 double bed', '')
# clean_only_combined_by_index(raw_data, 952424, 'share dormitory mix', '')
# add_label_and_clean_by_index(raw_data, 1197171, 'bdc_more bdt_double rmf_extra_bedding', 'special group package 5 double twin 1 full', '')
# add_label_and_clean_by_index(raw_data, 1208532, 'bdc_1 bdt_double', '6 double', '')
# add_label_and_clean_by_index(raw_data, 1218183, 'bdc_1 bdt_double rmt_room', '5 double garden', '')
# add_label_and_clean_by_index(raw_data, 1328171, 'bdc_more bdt_double_deck', 'dorm 5 double decker bed', '')
# add_label_and_clean_by_index(raw_data, 952424, 'bdc_more rmt_dorm', 'double bed dorm single double occupancy 9 double', '')

# add_label_and_clean(raw_data, 'double|double bed','bdc_1 bdt_double')

# add_label_and_clean(raw_data, '1 twin','bdc_1 bdt_twin')
# add_label_and_clean(raw_data, '4 twin','bdc_4 bdt_twin')
# add_label_and_clean(raw_data, 'mountain view','rmv_good')
# add_label_and_clean(raw_data, 'full kitchen','rmf_full_kitchen')
# add_label_and_clean(raw_data, 'kitchenette','rmf_kitchenette')


# add_label_and_clean(raw_data, 'harbour view','rmv_good')
# add_label_and_clean(raw_data, 'balcony','rmf_balcony')
# add_label_and_clean(raw_data, 'family multiple bedrooms beach view beachfront','rmc_more rmt_bedroom rmv_better rml_direct_access')

# add_label_and_clean(raw_data, 'icon 36 twin','bdc_1 bdt_twin')
# add_label_and_clean(raw_data, '6 twin free welcome drink','bdc_1 bdt_twin prmf_welcome_item')
# add_label_and_clean(raw_data, 'comfort villa s 6 twin','bdc_6 bdt_twin rmt_villa')
# add_label_and_clean(raw_data, '20 twin supreme foxtel first floor','rmf_extra_bedding bdt_twin rmt_superior rml_loflr')
# add_label_and_clean(raw_data, 'house 8 twin','bdc_more bdt_twin rmt_residence')
# add_label_and_clean(raw_data, 'house 6 bedrooms','rmc_more rmt_bedroom rmt_residence')

# clean_only_combined_by_index(raw_data, 120882, 'main tower high zone 28th 3 twin', '')
# clean_label_by_index(raw_data, 120882, 'rml_loflr', 'rml_hiflr')
# add_label_and_clean_by_index(raw_data, 120882, 'bdc_2 rmt_room bdt_twin rml_hiflr', 'main tower high zone 28th 3 twin', '')
# add_label_and_clean(raw_data, '5 bedrooms 6 twin|6 bedrooms 8 twinxl|5 bedrooms 10 twin|5 bedrooms 6 twin','rml_loflr')

# add_label_and_clean(raw_data, 'kitchen','rmf_full_kitchen')
# add_label_and_clean(raw_data, 'terrace','rmf_terrace')
# add_label_and_clean(raw_data, 'panoramic','rmv_best')

# add_label_and_clean(raw_data, '\\d+ twin','bdc_more bdt_twin')
# add_label_and_clean_by_index(raw_data, 1053494, 'bdc_2 rmt_suite bdt_twin', 'suite 1 1 2twins', '')
# add_label_and_clean_by_index(raw_data, 1414520, 'bdc_1 rmt_room bdt_twin', 'triple 1 bed 1twin bed share bathroom', '')

 
# clean_only_combined_with_label(raw_data, 'bdt_twin', 'twin', '')
 

# clean_combined(raw_data, 'double','bdt_queen','double','')

# add_label_and_clean(raw_data, 'full','bdc_1 bdt_full')
# add_label_and_clean(raw_data, 'welcome drink','prmf_welcome_item')
# clean_only_combined(raw_data, 'public area', '')
# add_label_and_clean(raw_data, 'suite','rmt_suite')
# add_label_and_clean(raw_data, 'full board','prmf_full_board')
# add_label_and_clean(raw_data, 'pool view','rmv_good')
# add_label_and_clean(raw_data, 'mahoora classic half day safari','rmt_title rmt_standard rtf_acctivity_included')
# add_label_and_clean(raw_data, 'tatami area half open air bath','rmt_standard rmf_traditional rmf_special_bath')
# add_label_and_clean(raw_data, 'tatami area half & dinner restaurant sakurabou','rmt_standard rmf_traditional prmf_half_board')
# add_label_and_clean(raw_data, 'japanese western style half open air bath 1f','rmt_standard rmf_traditional rmf_special_bath')
# add_label_and_clean(raw_data, ' half bottle champagne visit prestigious house champagne 2 people ','rtf_acctivity_included')
# add_label_and_clean(raw_data, 'standard half tester bed','rmt_standard')


# add_label_and_clean_by_labeled(raw_data, 'rmt_suite|bdt_king|bdt_queen|bdt_twin|bdt_double|rmt_bedroom|bdt_full|rmt_apartment|rmt_residence', 'sofa bed', 'rmf_extra_bedding', '')
# clean_label_by_column_name(raw_data, 'room_name', 'NONSMOKE', 'rmf_smoking', 'rmf_no_smoking')

# add_label_and_clean(raw_data, '1 sofa bed \\d+ full','rmf_extra_bedding')

# add_label_and_clean(raw_data, '2bd sofa bed non','rmc_2 rmt_bedroom rmf_extra_bedding')
# add_label_and_clean(raw_data, '1bd sofa bed non','rmc_1 rmt_bedroom rmf_extra_bedding')

# add_label_and_clean(raw_data, 'nsk w sofa bed ada','rmc_1 rmt_room bdc_1 bdt_sofa_bed rmf_accessible_mobility rmf_no_smoking')
# add_label_and_clean(raw_data, 'sofa bed ada','rmc_1 rmt_room bdc_1 bdt_sofa_bed rmf_accessible_mobility')
# add_label_and_clean(raw_data, 'sofa bed ada','rmc_1 rmt_room bdc_1 bdt_sofa_bed rmf_accessible_mobility')
# add_label_and_clean(raw_data, 'triple two single bed sofa bed','rmc_1 rmt_room bdc_3 bdt_single_sofa_bed')

# add_label_and_clean(raw_data, 'river view|bay view|park view','rmv_better')
# add_label_and_clean(raw_data, 'standard triple air condition exist sofa bed','rmc_1 rmt_standard bdc_3 bdt_single_sofa_bed')

# add_label_and_clean(raw_data, 'free airport shuttle','rtf_free_transport')
# add_label_and_clean(raw_data, 'free airport shuttle','rtf_free_transport')


# add_label_and_clean(raw_data, '2 person sofa bed 1 full','rmc_1 rmt_room bdc_1 bdt_full_bed rmf_extra_bedding')
# add_label_and_clean(raw_data, 'triple sofa bed','rmc_1 rmt_room bdc_3 bdt_sofa_bed')
# add_label_and_clean(raw_data, 'single plus sofa bed','rmc_1 rmt_room bdc_2 bdt_single_sofa_bed')

# add_label_and_clean(raw_data, '2 single 2 person sofa bed','rmc_1 rmt_room bdc_4 bdt_single_sofa_bed')
# add_label_and_clean(raw_data, 'triple 2 single bed 1 sofa bed','rmc_1 rmt_room bdc_3 bdt_single_sofa_bed')
# add_label_and_clean(raw_data, '1 single bed 1 single bed 1 sofa bed','rmc_1 rmt_room bdc_2 bdt_single_sofa_bed')
# add_label_and_clean(raw_data, 'corner rm 1 kg bed sofa bed','rmc_1 rmt_room bdc_1 bdt_king rmf_extra_bedding rmt_key_word')

# add_label_and_clean(raw_data, 'quadruple sofa bed','rmc_1 rmt_room bdc_4 bdt_sofa_bed')
# add_label_and_clean(raw_data, 'superior sofa bed 3 4 people use','rmc_1 rmt_room bdc_more bdt_sofa_bed rmt_superior')



# add_label_and_clean(raw_data, 'executive corner 1 full','rmc_1 rmt_room bdc_1 bdt_full rmt_key_word')
# add_label_and_clean(raw_data, 'sofa bed add 2nd guest 7 8f|sofa bed add 2nd guest 9f','rmc_1 rmt_room rmf_extra_bedding')

# add_label_and_clean(raw_data, '1 person sofa bed 1 full','rmc_1 rmt_room bdc_1 rmf_extra_bedding')
# add_label_and_clean(raw_data, 'renovate superior','rmc_1 rmt_superior rmt_key_word')
# add_label_and_clean(raw_data, '2 single bed sofa bed','bdc_2 bdt_single_bed rmf_extra_bedding')
# add_label_and_clean(raw_data, 'allergy friendly','rmf_allergy_friendly')

# clean_combined(raw_data, 'modular bathroom toilet partition shower cabin|room+ bathroom wash place toilet partition','rvw_','view','')

# add_label_and_clean_by_labeled(raw_data, 'rmt_suite|bdt_king|bdt_queen|bdt_twin|bdt_double|rmt_bedroom|bdt_full|rmt_apartment|rmt_residence|rmt_cottage|rmt_bungalow|rmt_villa|rmf_extra_bedding', '2 full', 'rmf_extra_bedding', '')

# add_label_and_clean(raw_data, 'house','rmt_residence')
# add_label_and_clean(raw_data, 'private pool','rml_pool_spa_access')
# add_label_and_clean_by_labeled(raw_data, 'rmt_', 'cottage','rmt_cottage')
# add_label_and_clean(raw_data, 'cottage','rmt_cottage')
# add_label_and_clean(raw_data, 'bungalow','rmt_bungalow')
# add_label_and_clean(raw_data, 'villa','rmt_villa')
# add_label_and_clean(raw_data, 'chalet','rmt_title')
# add_label_and_clean(raw_data, 'family concierge','rmt_title')
# add_label_and_clean(raw_data, 'family concierge ocean front inclusive|family concierge ocean front master inclusive','rmt_title prmf_all_inclusive')


# add_label_by_column_name(raw_data, 'room_name', 'Concierge Room', 'rml_special_space_access')
# clean_only_combined_with_label(raw_data, 'rml_concierge_level', 'concierge', '')

# add_label_and_clean(raw_data, 'deluxe concierge \\d+ adults inclusive','rmt_deluxe rmt_title prmf_all_inclusive')
# add_label_and_clean(raw_data, 'concierge','rml_concierge_level')

# clean_only_combined_with_label(raw_data, 'rmt_residence', '\\d+ full', '')
# add_label_and_clean(raw_data, 'harbor view full harbour|harbor view full harbour','rmv_good')
# add_label_and_clean(raw_data, 'remodel full harbour|four season harbor view remodel full harbour','rmv_good rmt_title')

# add_label_and_clean_by_labeled(raw_data, 'rmt_suite|bdt_king|bdt_queen|bdt_twin|bdt_double|rmt_bedroom|bdt_full|rmt_apartment|rmt_residence|rmt_cottage|rmt_bungalow|rmt_villa|rmf_extra_bedding', 'full  specialty', 'rmf_extra_bedding', '')


# add_label_and_clean(raw_data, 'luxury \\d+ bedrooms 7 full','rmv_good rmt_bedroom rmc_more bdc_more bdt_full rmt_title')
# add_label_and_clean(raw_data, 'mini full','rmt_mini rmc_1 bdc_1 bdt_full')
# add_label_and_clean(raw_data, 'w full pull sofa|junior w full pull sofa','rmf_extra_bedding')
# add_label_and_clean(raw_data, '1st avenue full','rmv_best')
# add_label_and_clean(raw_data, 'full harbor junior','rmt_title rmv_good')

# add_label_and_clean(raw_data, 'family 2 triple + 8 people 3 full','rmf_extra_bedding')
# add_label_and_clean(raw_data, 'premium s ribeira vintage duplex 3 full','bdt_full bdc_3')
# add_label_and_clean(raw_data, 'premium s ribeira vintage duplex 3 full','rmt_residence bdt_king bdt_queen bdt_full bdt_twinbdc_more rmf_extra_bedding')
# clean_label_by_column_name(raw_data, 'labels', 'bdt_twinbdc_more', 'bdt_twinbdc_more', 'bdt_twin bdc_more')
# add_label_and_clean(raw_data, 'full ocean front','rml_direct_access')


# clean_column_by_column_name(raw_data, 'labels', 'rmv_better', 'combined', 'full', '')


# add_label_and_clean_by_index(raw_data, 406027, 'bdc_1 bdt_full rmv_standard', '1 fullroom exterior view', '')
# add_label_and_clean_by_index(raw_data, 306329, 'bdc_1 bdt_full', '%d full', '')
# clean_only_combined_by_index(raw_data, 306328, '5 full', '')
# clean_only_combined_by_index(raw_data, 306329, '6 full', '')

# add_label_and_clean_if_labels_is_null(raw_data, 'full|1 full|one full', 'bdc_1 bdt_full', 'full|1 full|one full', '')


# add_label_and_clean(raw_data, 'deluxe  s 8 full','rmt_delux bdc_more bdt_full')
# add_label_and_clean(raw_data, 'full deluxe','rmt_delux bdc_1 bdt_full')
# add_label_and_clean(raw_data, 'superior full','rmt_superior bdc_1 bdt_full')
# add_label_and_clean(raw_data, 'full|full s','bdc_1 bdt_full')

# add_label_and_clean(raw_data, 'deluxe full','bdc_1 rmt_deluxe bdt_full')
# add_label_and_clean(raw_data, 'share bathroom 08 full','bdc_1 bdt_full')

### single room
# add_label_and_clean(raw_data, 'plus 1 single bed','rmf_extra_bedding')
# add_label_and_clean(raw_data, '\& single','rmf_extra_bedding')
# add_label_and_clean(raw_data, '\+ 1 single|plus 1 single|plus 2 single bed|\+ 2 single bed|\+ 2 single','rmf_extra_bedding')
# add_label_and_clean_if_labels_is_null(raw_data, '1 single bed|1 single|one single bed|one single', 'bdc_1 bdt_sinlge', '1 single bed|1 single|one single bed|one single', '')
# add_label_and_clean_if_labels_is_null(raw_data, 'single', 'bdc_1 bdt_sinlge', 'single', '')
# add_label_and_clean_if_label_not_present(raw_data, 'bdc_', 'single', 'bdc_1 bdt_single', '')



# add_label_and_clean_if_label_not_present(raw_data, 'bdc_', '1 single bed|1 single|one single bed|one single', 'bdc_1 bdt_sinlge', '')
# add_label_and_clean_if_label_not_present(raw_data, 'bdt_king', 'single superior', 'bdc_1 bdt_sinlge', '')

# clean_label_by_column_name(raw_data, 'room_name', 'Twin \(two single sized beds\)', 'bdt_twin', 'bdt_single')
# clean_label_by_column_name(raw_data, 'labels', 'bdt_sinlge', 'bdt_sinlge', 'bdt_sinlge')
# add_label_by_column_name(raw_data, 'room_name', 'Single Bed Room', 'rmt_bedroom rmc_1')
# clean_column_by_column_name(raw_data, 'room_name', 'Single Deluxe King Bed', 'combined', 'single', '')

# clean_label_by_column_name(raw_data, 'combined', 'single bed 2 adults', 'bdc_1 bdt_double', 'bdc_2 bdt_single')
# clean_label_by_column_name(raw_data, 'combined', 'single bed 3 adults', 'bdc_1 bdt_double', 'bdc_3 bdt_single')


# add_label_and_clean(raw_data, 'one single bed|single bed|single','bdc_1 bdt_single')
# clean_column_by_column_name(raw_data, 'room_name', 'Queen Single|SINGLE QUEEN SIZE BED|Single-queen', 'combined', 'single', '')
# clean_only_combined_with_label(raw_data, 'bdt_single', 'single', '')


# add_label_and_clean(raw_data, 'plus single','rmf_extra_bedding')
# add_label_and_clean(raw_data, 'single good 2 persons','bdc_2 bdt_sinlge')
# add_label_and_clean(raw_data, 'rustic single futon','rmt_title')
# add_label_and_clean(raw_data, 'private 4 single','bdc_4 bdt_capsule')
# add_label_and_clean(raw_data, 'family 8 2 subrooms 6 single','bdc_more bdt_sinlge rmc_2 rmt_room')
# add_label_and_clean(raw_data, 'family jet tub 3 single|connect 8 6 single \&|7 people family 5 single','rmf_extra_bedding')
# add_label_and_clean(raw_data, 'nyah private 5 single bed','rmc_5 rmt_bedroom rmf_extra_bedding')
# add_label_and_clean(raw_data, 's 4 single|4 single|executive 4 single','bdt_sinlge bdc_4')
# add_label_and_clean(raw_data, 'barn 3 single','bdt_sinlge bdc_3 rmt_title')
# add_label_and_clean(raw_data, '5 bedroom  5 single','bdt_sinlge bdc_more rmt_bedroom rmc_more')
# add_label_and_clean(raw_data, '\\d+ single','rmf_extra_bedding')

# add_label_and_clean_if_label_not_present(raw_data, 'bdc_', '5 single bed', 'bdc_more bdt_single', '')

# add_label_and_clean(raw_data, 'executive execuplus 2single bed free valet park','rmt_title bdc_2 bdt_single rtf_free_parking')
# add_label_and_clean(raw_data, 'romantic single room|business class single bed','rmt_title')



# two single bed|2 single bed
# add_label_if_labels_is_null(raw_data, 'single c', 'bdc_1 bdt_sinlge')
# clean_only_combined_with_label(raw_data, 'bdt_sinlge', 'single', '')

# add_label_and_clean_by_labeled(raw_data, 'bdc_1|bdc_2|bdc_3|bdc_4|bdc_more|bdt_', 'full', 'rmf_extra_bedding', '')

#credit

# add_label_and_clean(raw_data, 'cabana b \& b','rmt_title')


# add_label_and_clean(raw_data, 'free pick|free pickup','rtf_free_transport')

# add_label_if_labels_is_null(raw_data, 'beach front', 'rvw_best rml_direct_access')
# add_label_and_clean(raw_data, 'free valet park','rtf_free_parking')
# add_label_and_clean(raw_data, 'free water park entry','rtf_other_item_or_credit_included')

# add_label_and_clean(raw_data, 'arex ticket pkg airport rail ticket','rtf_other_item_or_credit_included')
# clean_only_combined(raw_data, '3rd night free', '')
# add_label_and_clean(raw_data, 'vip airport fast track welcome gift','prmf_welcome_item')

# clean_label_by_column_name(raw_data, 'room_name', 'No Breakfast', 'prmf_free_breakfast', '')
# clean_label_by_column_name(raw_data, 'room_name', 'No Airport', 'rtf_free_transport', '')
# clean_label_by_column_name(raw_data, 'room_name', 'Airport Parking Not Included', 'rtf_other_item_or_credit_included', '')

# add_label_and_clean(raw_data, 'premium','rmt_premium')
# add_label_and_clean(raw_data, 'superior','rmt_superior')
# add_label_and_clean(raw_data, 'studio','rmt_studio')

# add_label_and_clean(raw_data, 'panorama kng ste','rmv_best bdc_1 bdt_king rmt_suite')
# add_label_and_clean(raw_data, 'alii resort vw kng bd','rmv_good bdc_1 bdt_king')
# add_label_and_clean(raw_data, 'ada shwr kng lrg','rmf_accessible_mobility bdc_1 bdt_king')
# add_label_and_clean(raw_data, 'corner kng larger guest s','rmt_room rmc_1 bdc_1 bdt_king rmt_title')
# add_label_and_clean(raw_data, '2 br  2 kng plus sleeper sofa nonsm','rmt_bedroom rmc_2 bdc_2 bdt_king rmf_extra_bedding rmf_no_smoking')
# add_label_and_clean(raw_data, 'ada kng tub sfbd|ada kng tb|ada spec kng tub','rmf_accessible_mobility bdc_1 bdt_king')
# add_label_and_clean(raw_data, '2 kng w rollin shwr','bdc_2 bdt_king')
# add_label_and_clean(raw_data, 'junior kng inclusive','bdc_1 bdt_king prmf_all_inclusive rmt_title')

# add_label_and_clean(raw_data, 'kng bed mtg rm ste conf rm poolview','bdc_1 bdt_king rmt_suite rmv_standard')
# add_label_and_clean(raw_data, 'kng exec old city non|junior kng','bdc_1 bdt_king rmt_title')
# add_label_and_clean(raw_data, ' adults child|adults children ','bdc_1 bdt_king')
# add_label_and_clean(raw_data, 'kng  sleeper non','bdc_1 bdt_king rmf_extra_bedding')

# add_label_and_clean(raw_data, '2 qn bed|2qn bed|two qn bed','bdt_queen bdc_2')
# add_label_and_clean(raw_data, '1 qn bed|1qn bed|one qn bed','bdt_queen bdc_1')


# clean_column_by_column_name(raw_data, 'room_name', 'No view', 'combined', 'view', '')
# clean_column_by_column_name(raw_data, 'room_name', 'No view', 'labels', 'rmv_standard|rmv_good', '')

# add_label_by_column_name(raw_data, 'value_adds', 'all-inclusive', 'prmf_all_inclusive')
# clean_only_combined(raw_data, '\\d+ bathroom freestanding unit', '')
# add_label_and_clean(raw_data, 'ptl oceanvw','rmv_good')
# add_label_and_clean(raw_data, 'triple view','rmv_standard bdc_3')
# add_label_and_clean(raw_data, 'tatami area view','rmf_traditional')
# add_label_and_clean(raw_data, 'tub view|town view','rmv_standard')
# add_label_and_clean(raw_data, 'island view|guest skyline view s','rmv_good')
# add_label_and_clean(raw_data, 'view track view|guest fountain view|tower view lg styler','rmv_good')
# add_label_and_clean(raw_data, 'hammock view|atrium view|waterfall view|acropolis view|parliament view guest parliament view s','rmv_good')
# add_label_and_clean(raw_data, 'corner view','rmv_standard')
# add_label_and_clean(raw_data, 'cty view ste','rmv_standard rmt_suite')
# add_label_and_clean(raw_data, 'ocean front guest ocean front view|ocean front','rmv_good rml_direct_access')


# add_label_and_clean(raw_data, 'extend eddb guest','rmt_title')
# add_label_and_clean(raw_data, 'two db|2db|2 db','bdt_double bdc_2')
# add_label_and_clean(raw_data, 'one db|1db|1 db','bdt_double bdc_1')
# add_label_and_clean(raw_data, 'qb db','bdt_double bdt_queen bdc_1')
# add_label_and_clean(raw_data, ' db|db ','bdt_double bdc_1')


# add_label_and_clean(raw_data, '3bdrm','rmc_3 rmt_bedroom')
# add_label_and_clean(raw_data, '2bdrm','rmc_2 rmt_bedroom')
# add_label_and_clean(raw_data, '1bdrm','rmc_1 rmt_bedroom')

# add_label_and_clean(raw_data, '\[ special pkg \] beer sauna 2 choose 1 upon check|special pkg beers sauna','rtf_other_item_or_credit_included')
# add_label_and_clean(raw_data, 'allergy free','rmf_allergy_friendly')
# add_label_and_clean(raw_data, 'kg w poolvw','bdt_king bdc_1 rmv_standard')
# add_label_and_clean(raw_data, 'mailani ocean kg tower level','bdt_king bdc_1 rmt_title rml_special_flr')
# add_label_and_clean(raw_data, 'ste | ste','rmt_suite')
# add_label_and_clean(raw_data, 'free upgrade|rd night free','prmf_promotional')

# rmf_allergy_friendly to free_internet

# remove free if free in labels
# clean_only_combined_with_label(raw_data, 'standard', 'rmt_suite', '')

# add_label_and_clean(raw_data, '1bd ','bdc_1')
# add_label_and_clean(raw_data, '2bd ','bdc_2')
# add_label_and_clean_by_labeled(raw_data, 'rmt_premium|rmt_delux|rmt_superior|rmt_studio|rmt_parlor|rmt_suite|rmt_club|rmt_room|rmt_villa|rmt_penthouse|rmt_cabin|rmt_cottage|rmt_residence|rmt_apartment|rmt_bungalow|rmt_pod', 'executive', 'rmt_title', '')
# clean_only_combined(raw_data, 'non side','')
# add_label_and_clean(raw_data, 'refrigerator','rmf_frig')
# add_label_and_clean(raw_data, 'free 1 way','rtf_free_transport')
# add_label_and_clean(raw_data, 'grand new club floor free','rmt_grand rml_special_flr')
# add_label_and_clean(raw_data, '2 3 bed 1k2sgl kitch cabltv freepa','bdt_king bdt_single rmc_2 rmt_room bdc_3')
# add_label_and_clean(raw_data, '3 5 bed twnhse 1k4sgl kitch 50inlc freepa','bdt_king bdt_single rmc_2 rmt_room bdc_more')
# add_label_and_clean(raw_data, '2 3 bed exec 1k2sgl kitch cabltv freepa','bdt_king bdt_single rmc_2 rmt_room bdc_3 rmt_title')
# add_label_and_clean(raw_data, 'free 02 ways transfer','rtf_free_transport')
# add_label_and_clean(raw_data, 'japanese tatami mat','rmf_traditional')
# add_label_and_clean(raw_data, 'mrna fam ada','rmf_accessible_mobility rmt_title')
# add_label_and_clean(raw_data, ' view|view ','rmv_standard')

# clean_label_by_column_name(raw_data, 'labels', 'rmv_non', 'rmv_non', '')
# add_label_and_clean(raw_data, 'non smk|w non smkg|non smkg|nonsmo','rmf_no_smoking')
# add_label_and_clean(raw_data, 'non ac|non c','rmf_no_smoking') this is wrong
# add_label_and_clean(raw_data, 'non som|non sm|nonsm','rmf_no_smoking')
# add_label_and_clean(raw_data, 'nonrefundable|non ref|non refund','rtf_nrf')


# clean_label_by_column_name(raw_data, 'room_name', 'Non Pet Friendly|Non-Pet Friendly', 'rmf_pet_friendly', '')
# clean_label_by_column_name(raw_data, 'room_name', 'roman canon', 'rmt_residence', 'rmt_title rmt_penthouse')


# add_label_and_clean(raw_data, 'semi non mo annex extra 3rd adult','rmf_no_smoking')
# add_label_and_clean(raw_data, 'animal accept peche mignon','rmf_pet_friendly rmt_title')
# add_label_and_clean(raw_data, 'dubai','rmv_good')

# add_label_and_clean(raw_data, 'paul dubois','rmt_title')
# add_label_and_clean(raw_data, 'dublex mansion s|dublex','rmt_apartment')
# clean_label_by_column_name(raw_data, 'room_name', 'Non Breakfast', 'prmf_free_breakfast', '')
# clean_label_by_column_name(raw_data, 'room_name', 'Non- Pool View', 'rmv_good', 'rmv_non')



# add_label_by_column_name(raw_data, 'room_name', 'Non Air-Conditioning', 'rmf_no_ac')
# clean_column_by_column_name(raw_data, 'labels', 'rmf_no_smoking', 'combined', 'non', '')

# add_label_and_clean(raw_data, 'pualeilani resort fee|uxurious resort fee','rmt_title rtf_fee_included')
# add_label_and_clean(raw_data, 'resort fee hide charge park fee', 'rtf_fee_included rtf_free_parking')

# add_label_and_clean(raw_data, 'condo ferienwohnungurlaubsgluck', 'rmt_apartment')
# add_label_and_clean(raw_data, 'traditional self contain 2 brms 1 sgl verandah', 'rmc_2 rmt_bedroom bdt_double bdt_single')
# add_label_and_clean(raw_data, '43 1 sgl', 'bdc_4 bdt_single')


# add_label_by_column_name(raw_data, 'room_name', 'Free Spa Access', 'rtf_other_item_or_credit_included')
# add_label_by_column_name(raw_data, 'room_name', 'All Inclusive', 'prmf_all_inclusive')
# clean_label_by_column_name(raw_data, 'room_name', 'nonsmoke', 'rmf_smoking', 'rmf_no_smoking')
# clean_column_by_column_name(raw_data, 'labels', 'rmf_no_smoking', 'combined', ' non|non ', '')
# clean_column_by_column_name(raw_data, 'room_name',  'Non Seaview|Non-Seaview|Non-sea|Non-View|Non-Ocean View|Non sea view|Non Lake View', 'combined', 'non', '')

# clean_column_by_column_name(raw_data, 'room_name', 'All Inclusive', 'combined', 'inclusive|all', '')
         
# clean_only_combined(raw_data, '.*hors.*','')
# clean_label_by_column_name(raw_data, 'room_name', 'Twin 1 DBL', 'bdt_twin', 'rmt_twin')
# add_label_and_clean(raw_data, 'traditional self contain 2 brms 1 sgl verandah', 'rmc_2 rmt_bedroom rmf_extra_bedding')
# add_label_and_clean(raw_data, 'private w sgl superimpose', 'rmf_extra_bedding')
# add_label_and_clean(raw_data, 'stnd twsglb cteamk flatv frewi', 'rmf_free_internet rmf_extra_bedding')
# add_label_and_clean(raw_data, 'sofa|sleeper', 'rmf_extra_bedding')
# add_label_and_clean(raw_data, 'without airconditioning', 'rmf_no_ac')


# clean_label_by_column_name(raw_data, 'room_name', 'Family Room Twin \+2 \(4 Lits Sgl\)', '\+2 4 lits sgl', '')
    
# clean_label_by_column_name(raw_data, 'room_name', 'Smoke Free|Smoke\-Free', 'rmf_smoking', 'rmf_no_smoking')
# clean_column_by_column_name(raw_data, 'labels', 'rmf_accessible', 'combined', ' ada|ada ', '')
# remove_dup_labels(raw_data)

#raw_data.loc[(raw_data['combined'].str.contains('ada')) & (pd.notnull(raw_data['combined']))]
# raw_data.loc[(raw_data['combined'].str.contains(' ada|ada ')) & (pd.notnull(raw_data['combined']))]
# raw_data.loc[(raw_data['combined'].str.contains('open|air')) & (pd.notnull(raw_data['combined']))]
# raw_data.loc[(raw_data['combined'].str.contains('sofa bed')) & (pd.notnull(raw_data['combined']))]


# raw_data.loc[(raw_data['labels'].str.contains('rtf_single_use'))]


# selected_categories = [
#  'rmt_mini', 'rmt_premium', rmt_standard', 'rmt_delux', 'rmt_superior', 'rmt_grand', 'rmt_studio', 'rmt_parlor', 'rmt_non_suite', 'rmt_suite', 'rmt_bedroom', 'rmt_club', 'rmt_twin', 'rmt_triple', 'rmt_quadruple', 'rmt_room','rmt_villa','rmt_penthouse', 'rmt_apartment', 'rmt_dorm', 'rmt_key_word', 'rmt_cabin', 'rmt_cottage','rmt_residence', 'rmt_title', 'rmt_bungalow', 'rmt_pod',
#  'rmc_1', 'rmc_2', 'rmc_3', 'rmc_4', 'rmc_more',
#  'bdt_double', 'bdt_twin', 'bdt_queen', 'bdt_king','bdt_single','bdt_sofa_bed','bdt_single_sofa_bed','bdt_single_bed','bdt_full_bed','bdt_bunk_bed',
#  'bdc_1', 'bdc_2', 'bdc_3', 'bdc_4', 'bdc_more',
#  'prmf_all_inclusive', 'prmf_all_inclusive_lite', 'prmf_all_inclusive_semi','prmf_welcome_item','rtf_free_parking',
#  'rmv_standard', 'rmv_good', 'rmv_better', 'rmv_best','rmv_non',
#  'rml_hiflr', 'rml_loflr', 'rml_loft', 'rml_tower', 'rml_pool_spa_access','rml_direct_access','rml_special_space_access','rml_concierge_level','rml_lmited_or_no_access', 'rml_special_flr', 'rml_garden_area','rml_annex_building',
#  'rmf_accessible', 'rmf_smoking', 'rmf_no_smoking', 'rmf_full_kitchen', 'rmf_kitchenette', 'rmf_balcony', 'rmf_terrace', 'rmf_extra_bedding','rmf_mini_frig','rmf_frig','rmf_microwave','rmf_allergy_friendly','rmf_single_use','rmf_pet_friendly','rmf_traditional','rmf_free_internet',
#   'rmf_accessible_mobility', 'rmf_accessible', 'rmf_smoking', 'rmf_no_smoking','rmf_no_ac',
#  'rtf_free_cxl', 'rtf_nrf', 'rtf_fee_included', 'rtf_other_item_or_credit_included', 'rtf_free_transport', 'rtf_acctivity_included',
#  'prmf_free_breakfast', 'prmf_free_dinner', 'prmf_free_lunch', 'prmf_free_buffet', 'prmf_half_board', 'prmf_full_board', 'prmf_all_inclusive', 'prmf_all_inclusive_lite', 'prmf_all_inclusive_semi', 'prmf_welcome_item', 'prmf_promotional']