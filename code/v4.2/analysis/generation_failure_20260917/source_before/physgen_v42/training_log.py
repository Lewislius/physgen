"""Concise console progress and accounting for the routed training objectives."""
import sys


def loss_breakdown(metrics, weights, aux_alpha):
    """Report the scalar equivalent of the mixed gradients, before grad clipping.

    The actual backward passes stay separate. Alpha is a fixed gradient multiplier,
    and STRUCT's effective scalar weight multiplies mean(q_i * L_struct_i), not
    mean(L_struct_i). Preserve each sample's coefficient for exact inspection.
    """
    count = len(metrics)
    fm_active = [m["fm_active"] for m in metrics]
    temp_active = [m["temporal_active"] for m in metrics]
    q = [m["struct_noise_weight"] for m in metrics]
    fm = sum(m["weighted_fm"] for m in metrics)
    struct_nominal = sum(m["weighted_struct"] for m in metrics)
    temp = sum(m["weighted_temp"] for m in metrics)
    struct = aux_alpha * struct_nominal
    terms = dict(
        fm=dict(active=any(fm_active), base_weight=weights["fm"],
                effective_weight=weights["fm"] if any(fm_active) else 0.,
                raw_mean=sum(m["fm"] for m in metrics) / count if any(fm_active) else None,
                contribution=fm,
                micro_weights=[weights["fm"] / count if active else 0. for active in fm_active],
                micro_contributions=[m["weighted_fm"] for m in metrics]),
        struct=dict(active=True, base_weight=weights["struct"], aux_alpha=aux_alpha,
                    effective_weight=weights["struct"] * aux_alpha,
                    raw_mean=sum(m["struct"] for m in metrics) / count,
                    noise_weights=q,
                    # Derive from the tensors used for backward, including their rounding.
                    noise_weighted_mean=struct_nominal / weights["struct"],
                    nominal_contribution=struct_nominal, contribution=struct,
                    micro_weights=[weights["struct"] * noise * aux_alpha / count for noise in q],
                    micro_contributions=[aux_alpha * m["weighted_struct"] for m in metrics]),
        temp=dict(active=any(temp_active), base_weight=weights["temp"],
                  effective_weight=weights["temp"] if any(temp_active) else 0.,
                  raw_sum=sum(m["temp"] for m in metrics) if any(temp_active) else None,
                  contribution=temp,
                  micro_weights=[weights["temp"] if active else 0. for active in temp_active],
                  micro_contributions=[m["weighted_temp"] for m in metrics]),
    )
    return dict(loss_fm=fm, loss_struct_nominal=struct_nominal, loss_struct=struct,
                loss_temp=temp, loss_total=fm + struct + temp, loss_terms=terms)


def format_losses(values):
    terms = values["loss_terms"]
    parts = []
    for name in ("fm", "struct", "temp"):
        term = terms[name]
        if not term["active"]:
            parts.append(f"{name.upper()}(off,w=0,add=0)")
        elif name == "struct":
            parts.append(f"STRUCT(qL={term['noise_weighted_mean']:.6g},"
                         f"w={term['base_weight']:.6g}*{term['aux_alpha']:.6g}"
                         f"={term['effective_weight']:.6g},add={term['contribution']:.6g})")
        else:
            parts.append(f"{name.upper()}(w={term['effective_weight']:.6g},"
                         f"add={term['contribution']:.6g})")
    if "aux_grad_share" in values:
        parts.append(f"STRUCT.grad_share={values['aux_grad_share']:.2%}/{values['aux_max_share']:.0%}")
    return " | ".join(parts) + f" | TOTAL={values['loss_total']:.6g}"


class TrainingProgress:
    """Refresh one line on a terminal; emit small progress lines in job logs."""
    def __init__(self, total, stream=None):
        self.total = total
        self.stream = sys.stdout if stream is None else stream
        self.interactive = self.stream.isatty()
        self.width = 0

    def _write(self, line, complete=False):
        if self.interactive:
            padding = " " * max(0, self.width - len(line))
            self.stream.write("\r" + line + padding + ("\n" if complete else ""))
            self.width = 0 if complete else len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def start_step(self, step, micro_total):
        self.step, self.micro_total = step, micro_total
        self.advance(0)

    def _prefix(self, completed):
        filled = 16 * completed // self.micro_total
        bar = "#" * filled + "-" * (16 - filled)
        return f"step {self.step}/{self.total} | micro [{bar}] {completed}/{self.micro_total}"

    def advance(self, completed):
        # The final 8/8 line is emitted after optimizer.step with final loss weights.
        if completed < self.micro_total:
            self._write(self._prefix(completed))

    def finish_step(self, values):
        self._write(self._prefix(self.micro_total) + " | " + format_losses(values), complete=True)

    def close(self):
        if self.interactive and self.width:
            self.stream.write("\n")
            self.stream.flush()
            self.width = 0

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
