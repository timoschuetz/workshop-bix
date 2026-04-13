# Challenge A — Golden Batch Detective (Manufacturing Context)

## Story / Kontext
In einer biopharmazeutischen Produktion werden viele Prozesssignale während eines **Batch-Laufs** erfasst (z. B. Temperatur, Druck, pH, Rührerdrehzahl, Zuführungen). Ziel ist ein möglichst **robuster, reproduzierbarer Ablauf** mit hoher Qualität und Ausbeute. Der Begriff **"Golden Batch"** beschreibt einen idealen Lauf, der als **Benchmark** für spätere Batches dient. 

Im Hackathon bekommt ihr synthetische, aber realistisch strukturierte Batch-Zeitreihen. Eure Aufgabe ist es, aus "guten" historischen Batches ein Golden-Profil abzuleiten und neue Batches **früh** als "on track" oder "abweichend" zu erkennen.

## Aufgabenstellung
Entwickelt einen Agentic-AI-gestützten Prototypen, der:
1. **Golden Batch Profil** lernt (z. B. pro Phase/über Prozessfortschritt).
2. Für einen neuen Batch **Abweichungen frühzeitig** erkennt (möglichst bevor der Batch fertig ist).
3. Eine **Treiberanalyse** liefert: Welche Variablen/Phasen tragen am stärksten zur Abweichung bei?
4. Einen **kurzen Operator-Report** generiert (2–6 Sätze, verständlich, mit Handlungsempfehlung).

## Input-Daten (Dummy)
- `caseA_timeseries.csv`: Zeitreihen pro Batch (t_pct, Phase, Temperatur, Druck, pH, RPM, Feeds)
- `caseA_batches.csv`: Batch-Metadaten und Outcome (quality_pass, yield_kg, Label is_anomalous)

## Agentic AI Idee (optional)
- Agent 1: Data Prep (Phasen, Normalisierung)
- Agent 2: Monitoring/Anomaly
- Agent 3: RCA + Report
