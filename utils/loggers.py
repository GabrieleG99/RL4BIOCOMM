from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class MetricsStore:
    """
    Archivio centralizzato e algoritmo-agnostico per le metriche di training.

    Supporta due livelli di metriche:
      - **scalari globali** (es. val_acc, avg_reward, traj_len)
      - **metriche per-agente** (es. loss, entropy, avg_return per ciascun agente)

    Le metriche vengono registrate dinamicamente: non serve dichiarare in
    anticipo le chiavi, quindi funziona con qualsiasi algoritmo.

    Uso tipico
    ----------
    >>> store = MetricsStore()
    >>> store.log("val_acc", 0.87)                              # metrica globale
    >>> store.log("loss", 1.23, agent_id="agent_0")             # metrica per-agente
    >>> store.log_dict({"entropy": 0.5, "critic_loss": 0.3}, agent_id="agent_0")
    >>> store.log_for_agents(["agent_0", "agent_1"], {"loss": [1.1, 1.3]})
    >>> store["val_acc"]               # -> [0.87]
    >>> store.agent("agent_0")         # -> {"loss": [1.23, 1.1], "entropy": [0.5], ...}
    """

    def __init__(self) -> None:
        # Metriche globali:  key -> list[float]
        self._global: dict[str, list] = defaultdict(list)
        # Metriche per-agente:  agent_id -> key -> list[float]
        self._per_agent: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        # Metadati opzionali (iperparametri, info run, …)
        self._metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #  Logging
    # ------------------------------------------------------------------ #
    def log(self, key: str, value: Any, *, agent_id: Optional[str] = None,
            step: Optional[int] = None) -> None:
        """
        Registra un singolo valore.

        Parameters
        ----------
        key : str
            Nome della metrica (es. "val_acc", "loss").
        value : Any
            Valore da registrare (scalare, array, …).
        agent_id : str, optional
            Se fornito, la metrica viene archiviata per quell'agente.
        step : int, optional
            Step esplicito (per ora non usato, ma utile per estensioni future).
        """
        if agent_id is not None:
            self._per_agent[agent_id][key].append(value)
        else:
            self._global[key].append(value)

    def log_dict(self, d: dict[str, Any], *, agent_id: Optional[str] = None) -> None:
        """Registra più metriche da un dizionario in un colpo solo."""
        for key, value in d.items():
            self.log(key, value, agent_id=agent_id)

    def log_for_agents(self, agent_ids: Sequence[str], metrics: dict[str, Sequence[Any]]) -> None:
        """
        Registra metriche per più agenti in un colpo solo.

        Utile quando l'algoritmo produce metriche aggregate su un gruppo di agenti
        (es. tutti quelli che condividono una policy) e si vuole distribuire
        il risultato per-agente.

        Parameters
        ----------
        agent_ids : Sequence[str]
            Lista degli agent_id a cui associare le metriche.
        metrics : dict[str, Sequence[Any]]
            Dizionario chiave -> lista di valori. Se la lista ha lo stesso
            numero di elementi di agent_ids, ogni valore viene assegnato
            al rispettivo agente. Se ha un solo elemento (o è uno scalare),
            lo stesso valore viene replicato per tutti gli agenti.

        Esempio
        -------
        >>> store.log_for_agents(
        ...     ["agent_0", "agent_1", "agent_2"],
        ...     {"avg_return": [0.5, 0.6, 0.7], "loss": 1.23}
        ... )
        """
        for key, values in metrics.items():
            if isinstance(values, (list, tuple, np.ndarray)):
                if len(values) == len(agent_ids):
                    for aid, v in zip(agent_ids, values):
                        self._per_agent[aid][key].append(
                            v.item() if isinstance(v, (np.generic, np.ndarray)) else v
                        )
                elif len(values) == 1:
                    v = values[0]
                    v = v.item() if isinstance(v, (np.generic, np.ndarray)) else v
                    for aid in agent_ids:
                        self._per_agent[aid][key].append(v)
                else:
                    raise ValueError(
                        f"Metrica '{key}': lunghezza {len(values)} non compatibile "
                        f"con {len(agent_ids)} agenti. Deve essere uguale o 1."
                    )
            else:
                # Scalare → replica per tutti gli agenti
                for aid in agent_ids:
                    self._per_agent[aid][key].append(values)

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    # ------------------------------------------------------------------ #
    #  Accesso
    # ------------------------------------------------------------------ #
    def __getitem__(self, key: str) -> list:
        """Accedi a una metrica globale per nome."""
        return self._global[key]

    def __contains__(self, key: str) -> bool:
        return key in self._global

    def get(self, key: str, default: Any = None) -> Any:
        return self._global.get(key, default)

    def agent(self, agent_id: str) -> dict[str, list]:
        """Restituisce tutte le metriche di un singolo agente."""
        return dict(self._per_agent[agent_id])

    def agent_metric(self, agent_id: str, key: str) -> list:
        """Restituisce una metrica specifica di un agente."""
        return self._per_agent[agent_id][key]

    def agents_metric(self, key: str, agent_ids: Optional[Sequence[str]] = None) -> dict[str, list]:
        """
        Restituisce una specifica metrica per più agenti.

        Parameters
        ----------
        key : str
            Nome della metrica.
        agent_ids : Sequence[str], optional
            Lista di agenti. Se None, restituisce per tutti.
        """
        ids = agent_ids or self.agent_ids
        return {aid: self._per_agent[aid][key] for aid in ids if key in self._per_agent[aid]}

    @property
    def global_keys(self) -> list[str]:
        return list(self._global.keys())

    @property
    def agent_ids(self) -> list[str]:
        return list(self._per_agent.keys())

    def agent_keys(self, agent_id: str) -> list[str]:
        return list(self._per_agent[agent_id].keys())

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    # ------------------------------------------------------------------ #
    #  Ultimi N valori / best
    # ------------------------------------------------------------------ #
    def last(self, key: str, n: int = 1, *, agent_id: Optional[str] = None) -> Any:
        series = self.agent_metric(agent_id, key) if agent_id else self._global[key]
        if n == 1:
            return series[-1] if series else None
        return series[-n:]

    def best(self, key: str, mode: str = "max", *, agent_id: Optional[str] = None) -> tuple[int, Any]:
        """Restituisce (step, valore) del miglior risultato."""
        series = self.agent_metric(agent_id, key) if agent_id else self._global[key]
        fn = max if mode == "max" else min
        best_val = fn(series)
        best_step = series.index(best_val)
        return best_step, best_val

    # ------------------------------------------------------------------ #
    #  Export
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Esporta tutto come dizionario serializzabile."""
        return {
            "global": {k: list(v) for k, v in self._global.items()},
            "per_agent": {
                aid: {k: list(v) for k, v in metrics.items()}
                for aid, metrics in self._per_agent.items()
            },
            "metadata": self._metadata,
        }

    def to_dataframe(self, scope: str = "global") -> "pd.DataFrame":
        """
        Converte le metriche in un DataFrame pandas.

        Parameters
        ----------
        scope : str
            "global" per le metriche globali, oppure un agent_id.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas è richiesto per to_dataframe()")
        if scope == "global":
            return pd.DataFrame(dict(self._global))
        return pd.DataFrame(dict(self._per_agent[scope]))

    def save(self, path: str | Path) -> None:
        """Salva le metriche in formato JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, default=_convert, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "MetricsStore":
        """Carica le metriche da un file JSON."""
        with open(path) as f:
            data = json.load(f)
        store = cls()
        for k, v in data.get("global", {}).items():
            store._global[k] = v
        for aid, metrics in data.get("per_agent", {}).items():
            for k, v in metrics.items():
                store._per_agent[aid][k] = v
        store._metadata = data.get("metadata", {})
        return store

    def __repr__(self) -> str:
        g = len(self._global)
        a = len(self._per_agent)
        return f"MetricsStore(global_metrics={g}, agents={a})"