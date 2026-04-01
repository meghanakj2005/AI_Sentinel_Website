import pandas as pd
from sklearn.preprocessing import OneHotEncoder

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type", "level"
]

DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "mailbomb", "apache2", "processtable", "udpstorm", "worm"
}
PROBE_ATTACKS = {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"}
R2L_ATTACKS = {
    "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
    "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
    "snmpguess", "xlock", "xsnoop", "httptunnel"
}
U2R_ATTACKS = {"buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"}


def map_attack_category(attack_name: str) -> str:
    if attack_name == "normal":
        return "normal"
    if attack_name in DOS_ATTACKS:
        return "DoS"
    if attack_name in PROBE_ATTACKS:
        return "Probe"
    if attack_name in R2L_ATTACKS:
        return "R2L"
    if attack_name in U2R_ATTACKS:
        return "U2R"
    return "DoS"


def load_nsl_kdd(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, header=None, names=NSL_KDD_COLUMNS)


def add_attack_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["attack_class"] = df["attack_type"].apply(map_attack_category)
    return df


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame = None):
    """
    Based on your notebook:
    - OneHotEncode protocol_type and flag
    - Frequency encode service
    - Keep numeric columns
    """
    train_df = train_df.copy()
    test_mode = test_df is not None

    if test_mode:
        test_df = test_df.copy()

    cols_to_encode = ["protocol_type", "flag"]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(train_df[cols_to_encode])

    train_encoded = pd.DataFrame(
        ohe.transform(train_df[cols_to_encode]),
        columns=ohe.get_feature_names_out(cols_to_encode),
        index=train_df.index
    )

    freq_map = train_df["service"].value_counts(normalize=True)
    train_df["service_freq"] = train_df["service"].map(freq_map).fillna(0)

    train_final = pd.concat([train_df, train_encoded], axis=1)

    if not test_mode:
        return train_final, freq_map, ohe

    test_encoded = pd.DataFrame(
        ohe.transform(test_df[cols_to_encode]),
        columns=ohe.get_feature_names_out(cols_to_encode),
        index=test_df.index
    )
    test_df["service_freq"] = test_df["service"].map(freq_map).fillna(0)
    test_final = pd.concat([test_df, test_encoded], axis=1)

    return train_final, test_final, freq_map, ohe


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = ["protocol_type", "service", "flag", "attack_type", "level"]
    existing = [col for col in drop_cols if col in df.columns]
    return df.drop(columns=existing)


def align_to_expected_columns(df: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]