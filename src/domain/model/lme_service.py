import statsmodels.formula.api as smf
import pandas as pd


class LMEService:
    @staticmethod
    def fit_null_model(
        df: pd.DataFrame, response_col: str, group_col: str = "speaker_id"
    ):
        formula = f"{response_col} ~ 1"
        model = smf.mixedlm(formula, df, groups=df[group_col])
        result = model.fit(method="cg")

        sig2_u = result.cov_re.iloc[0, 0]
        sig2_eps = result.scale

        icc = sig2_u / (sig2_u + sig2_eps)
        return result, float(icc)

    @staticmethod
    def fit_full_model(
        df: pd.DataFrame, response_col: str, group_col: str = "speaker_id"
    ):
        formula = f"{response_col} ~ C(l1_status) * C(gender)"
        model = smf.mixedlm(formula, df, groups=df[group_col])
        result = model.fit(method="cg")
        return result
