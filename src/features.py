"""Versioned features computed solely from the observer's wire trace."""
import numpy as np

SCHEMA = "packet284+directional100-v2"


def schema(config):
    return SCHEMA if config.get("directional", False) else "packet284-v1"


def feature_contract(config):
    """Only settings consumed by the extractor; descriptive YAML fields are ignored.

    Keep this contract and the schema version synchronized with feature changes.
    Historical normalization/statistics labels documented hard-coded operations;
    removing those labels never changed the resulting numerical features.
    """
    def pick(values, *keys):
        return {key: values[key] for key in keys}

    size, timing = config["size"], config["timing"]
    result = dict(schema=schema(config), std_ddof=config["std_ddof"],
                  quantile_method=config["quantile_method"], size=pick(size,
                  "quantile_percentiles", "quantile_scale_bytes", "mean_std_scale_bytes",
                  "small_max_bytes", "large_min_bytes"), timing=pick(timing, "quantile_percentiles"))
    result["size"]["histogram"] = pick(size["histogram"], "first_edge_bytes",
                                       "last_edge_bytes", "bin_width_bytes")
    result["size"]["prefix"] = pick(size["prefix"], "length", "padding_bytes", "scale_bytes")
    result["size"]["transitions"] = pick(size["transitions"], "edges_bytes", "upper_edge_inclusive")
    result["timing"]["histogram"] = pick(timing["histogram"], "first_edge", "last_edge", "edge_count")
    if config.get("directional", False):
        result["observation_seconds"] = config["observation_seconds"]
    return result


def directional_features(packets, horizon):
    """100 direction, volume, burst and fixed-window features; direction is 0/1."""
    if not len(packets):
        return np.zeros(100)
    time = packets[:, 0].astype(float) / 1e9
    direction, size = packets[:, 1], packets[:, 2].astype(float)
    count = len(packets)
    output = [np.array([count, size.sum(), time[-1] - time[0],
                        np.mean(direction[1:] != direction[:-1]) if count > 1 else 0])]
    for side in (0, 1):
        values = size[direction == side]
        output.append(np.r_[len(values), values.sum(), values.mean(), values.std(),
                            np.quantile(values, [0.5, 0.9]), values.max()]
                      if len(values) else np.zeros(7))
        gaps = np.diff(time[direction == side])
        output.append(np.r_[gaps.mean(), gaps.std(), np.quantile(gaps, [0.5, 0.9])]
                      if len(gaps) else np.zeros(4))
    starts = np.r_[0, np.flatnonzero(np.diff(direction) != 0) + 1]
    lengths = np.diff(np.r_[starts, count])
    volumes = np.add.reduceat(size, starts)
    for side in (0, 1):
        selected = direction[starts] == side
        output.append(np.array([selected.sum(), lengths[selected].mean(),
                                lengths[selected].max(), volumes[selected].mean(),
                                volumes[selected].max()]) if selected.any() else np.zeros(5))
    prefix = np.zeros(32)
    prefix[:min(count, 32)] = (1 - 2 * direction[:32]) * size[:32]
    output.append(prefix)
    # Absolute observer-window bins: a delayed first packet must not reset time.
    bins = np.clip((8 * time / horizon).astype(int), 0, 7)
    for side in (0, 1):
        selected = direction == side
        output.extend([np.bincount(bins[selected], minlength=8),
                       np.bincount(bins[selected], weights=size[selected], minlength=8)])
    return np.concatenate(output)


def extract_features(packets, config):
    """Features of time-ordered packets [ns, direction, bytes], in a fixed window."""
    if not len(packets):
        return np.zeros(384 if config.get("directional", False) else 284, dtype=np.float32)
    sizes = packets[:, 2]
    gaps = np.diff(packets[:, 0]).astype(float) / 1e9
    size, timing = config["size"], config["timing"]
    histogram = size["histogram"]
    edges = np.arange(histogram["first_edge_bytes"], histogram["last_edge_bytes"] + 1,
                      histogram["bin_width_bytes"])
    mean, std = sizes.mean(), sizes.std(ddof=config["std_ddof"])
    quantiles = np.quantile(sizes, np.array(size["quantile_percentiles"]) / 100,
                            method=config["quantile_method"])
    prefix = np.full(size["prefix"]["length"], float(size["prefix"]["padding_bytes"]))
    prefix[:len(sizes)] = sizes[:len(prefix)]
    categories = np.digitize(sizes, size["transitions"]["edges_bytes"],
                             right=size["transitions"]["upper_edge_inclusive"])
    bins = len(size["transitions"]["edges_bytes"]) + 1
    transitions = np.bincount(bins * categories[:-1] + categories[1:], minlength=bins ** 2)
    size_features = np.r_[np.histogram(sizes, edges)[0] / len(sizes),
                          quantiles / size["quantile_scale_bytes"],
                          mean / size["mean_std_scale_bytes"], std / size["mean_std_scale_bytes"],
                          std / mean, np.mean(sizes <= size["small_max_bytes"]),
                          np.mean(sizes >= size["large_min_bytes"]),
                          prefix / size["prefix"]["scale_bytes"],
                          transitions / max(1, len(sizes) - 1)]
    if len(gaps):
        q = np.quantile(gaps, np.array(timing["quantile_percentiles"]) / 100,
                        method=config["quantile_method"])
        mean, std = gaps.mean(), gaps.std(ddof=config["std_ddof"])
        summary = np.r_[q, mean, std, std / mean if mean else 0.0, q[4] - q[2],
                        np.mean(gaps == 0), (packets[-1, 0] - packets[0, 0]) / 1e9, 0.0]
    else:
        summary = np.r_[np.zeros(len(timing["quantile_percentiles"]) + 6), 1.0]
    histogram = timing["histogram"]
    edges = np.linspace(histogram["first_edge"], histogram["last_edge"], histogram["edge_count"])
    counts = np.histogram(np.log10(1 + gaps * 1e9), edges)[0] / max(1, len(gaps))
    values = np.r_[size_features, summary, counts]
    if config.get("directional", False):
        values = np.r_[values, directional_features(packets, config["observation_seconds"])]
    return values.astype(np.float32)
