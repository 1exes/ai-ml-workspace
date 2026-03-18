import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
Text Klassifikation - Eigenen Classifier trainieren
=====================================================
TF-IDF + sklearn vs. BERT Fine-Tuning auf deutschen Texten.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# ============================================================
# 1. Deutschen Datensatz erstellen
# ============================================================
print("=" * 60)
print("1. DEUTSCHER TEXT-DATENSATZ")
print("=" * 60)

# 4 Kategorien, je 25+ Samples
texts_sport = [
    "Der FC Bayern hat gestern das Spiel gegen Dortmund gewonnen",
    "Beim Marathon in Berlin wurde ein neuer Rekord aufgestellt",
    "Die deutsche Nationalmannschaft bereitet sich auf die EM vor",
    "Im Tennis hat Alexander Zverev das Finale erreicht",
    "Der neue Trainer setzt auf eine offensive Spielstrategie",
    "Beim Schwimm-Weltcup gab es drei Goldmedaillen fuer Deutschland",
    "Die Bundesliga startet am Wochenende in die neue Saison",
    "Der Handballverein feiert den Aufstieg in die erste Liga",
    "Beim Formel-1-Rennen gab es einen spannenden Zweikampf",
    "Die Fussballspieler trainieren fuer das Champions-League-Spiel",
    "Olympische Spiele bringen neue Medaillen fuer deutsche Athleten",
    "Der Basketballverein verstaerkt sich mit einem amerikanischen Center",
    "Beim Biathlon gab es einen packenden Zielsprint",
    "Die Volleyball-Mannschaft gewinnt den Pokal zum dritten Mal",
    "Neue Transfergeruechte um den Stuermer des Vereins",
    "Der Eishockey-Spieler wechselt in die NHL nach Nordamerika",
    "Beim Leichtathletik-Meeting wurden drei Landesrekorde gebrochen",
    "Die Radsport-Tour fuehrt durch die Alpen ueber drei Paesse",
    "Fitness-Trend: Immer mehr Deutsche gehen regelmaessig ins Gym",
    "Der Box-Weltmeister verteidigt seinen Titel erfolgreich",
    "Beim Skifahren sorgte ein Sturz fuer Aufregung im Zielbereich",
    "Turnverein meldet Rekordzahlen bei den Neuanmeldungen",
    "Golf-Turnier zieht tausende Zuschauer an den Platz",
    "Surfer nutzen die hohen Wellen an der Nordseekueste",
    "Triathlon-Saison startet mit dem Ironman in Hamburg",
]

texts_tech = [
    "Apple stellt das neue iPhone mit verbesserter KI-Funktion vor",
    "Der neue Quantencomputer von Google erreicht 1000 Qubits",
    "Microsoft integriert Copilot in alle Office-Anwendungen",
    "Die neue Programmiersprache Mojo verspricht Python-Speed wie C",
    "Cybersecurity-Experten warnen vor neuer Ransomware-Welle",
    "Das Software-Update behebt kritische Sicherheitsluecken",
    "Kuenstliche Intelligenz revolutioniert die Medikamentenentwicklung",
    "Der neue Prozessor ist doppelt so schnell wie der Vorgaenger",
    "Cloud-Computing-Markt waechst um 30 Prozent gegenueber Vorjahr",
    "Open-Source-Projekt erreicht eine Million Downloads auf GitHub",
    "Virtual Reality wird in der Chirurgie-Ausbildung eingesetzt",
    "Blockchain-Technologie findet Anwendung in der Lieferkette",
    "Der Roboter kann jetzt komplexe Aufgaben selbststaendig loesen",
    "Neue 5G-Netze ermoeglichen Geschwindigkeiten von 10 Gigabit",
    "Entwickler nutzen KI-Tools fuer automatische Code-Generierung",
    "Smart-Home-Geraete werden durch Matter-Standard kompatibel",
    "Die Linux-Distribution bekommt ein neues Desktop-Design",
    "3D-Drucker fertigen jetzt auch Bauteile aus Metall",
    "Autonomes Fahren: Tesla und Waymo liefern sich ein Wettrennen",
    "Server-Ausfaelle legten mehrere Online-Dienste lahm",
    "Datenschutz: EU verschaerft Regeln fuer Tech-Konzerne",
    "Neues Framework vereinfacht die Web-Entwicklung erheblich",
    "Batterietechnologie ermoeglicht Smartphones mit einer Woche Laufzeit",
    "USB-C wird Pflicht fuer alle elektronischen Geraete in der EU",
    "Quantenverschluesselung macht Kommunikation abhoersicher",
]

texts_politik = [
    "Der Bundestag debattiert ueber das neue Klimaschutzgesetz",
    "Die Koalitionsverhandlungen zwischen den Parteien dauern an",
    "EU-Gipfel beschliesst neue Sanktionen gegen Russland",
    "Wahlkampf: Kandidaten stellen ihre Programme vor",
    "Der Aussenminister reist zu Verhandlungen nach Washington",
    "Neue Steuerreform soll den Mittelstand entlasten",
    "Protest gegen die geplante Rentenreform in mehreren Staedten",
    "Die Opposition fordert einen Untersuchungsausschuss",
    "Kommunalwahlen bringen ueberraschende Ergebnisse in Bayern",
    "Der Verteidigungsminister kuendigt Bundeswehr-Modernisierung an",
    "Fluechtlingspolitik bleibt zentrales Thema im Wahlkampf",
    "NATO-Treffen: Deutschland sagt hoehere Verteidigungsausgaben zu",
    "Bildungspolitik: Laender einigen sich auf digitale Standards",
    "Gesundheitsminister stellt Krankenhausreform vor",
    "Demonstration fuer mehr Klimaschutz vor dem Kanzleramt",
    "Die Gruenen fordern den schnelleren Ausstieg aus fossilen Energien",
    "Handelsabkommen mit Suedamerika steht vor dem Abschluss",
    "Innenministerin praesentiert neues Sicherheitskonzept",
    "Verfassungsgericht urteilt ueber umstrittenes Gesetz",
    "Regierung beschliesst Hilfspaket fuer Hochwasser-Opfer",
    "Lokalpolitik: Buergermeisterwahl in der Landeshauptstadt",
    "Parlamentarier debattieren ueber die Schuldenbremse",
    "Diplomatische Krise zwischen EU und China wegen Handelsstreit",
    "Parteivorsitzender tritt nach Wahlniederlage zurueck",
    "Bundespraesident haelt Rede zum Tag der Deutschen Einheit",
]

texts_wirtschaft = [
    "DAX erreicht neues Allzeithoch ueber 20.000 Punkte",
    "Volkswagen kuendigt Milliarden-Investition in E-Mobilitaet an",
    "Die Inflation sinkt auf 2,1 Prozent im Jahresvergleich",
    "Start-up aus Berlin sammelt 50 Millionen Euro Finanzierung",
    "Die EZB hebt den Leitzins um 0,25 Prozentpunkte an",
    "Immobilienpreise fallen in deutschen Grossstaedten",
    "Export-Rekord: Deutsche Wirtschaft waechst staerker als erwartet",
    "Fachkraeftemangel kostet die Wirtschaft Milliarden Euro jaehrlich",
    "Bitcoin steigt auf ueber 100.000 Dollar",
    "Siemens baut neues Werk fuer Kuenstliche Intelligenz in Bayern",
    "Einzelhandel meldet Umsatzrueckgang im Weihnachtsgeschaeft",
    "Mindestlohn steigt auf 13 Euro pro Stunde",
    "Insolvenzwelle: Zahl der Firmenpleiten steigt deutlich",
    "Deutsche Bank meldet hoechsten Gewinn seit zehn Jahren",
    "Energiepreise belasten mittelstaendische Unternehmen",
    "Handwerk sucht dringend Auszubildende und Fachkraefte",
    "Lieferkettenprobleme entspannen sich nach zwei Jahren Krise",
    "Boerse reagiert nervoes auf US-Zinsentscheidung",
    "Automobilindustrie investiert in Wasserstoff-Technologie",
    "Arbeitslosenquote faellt auf niedrigsten Stand seit Wiedervereinigung",
    "Gastgewerbe erholt sich langsam von den Pandemie-Folgen",
    "Halbleiter-Werk: Intel baut neue Fabrik in Magdeburg",
    "Mittelstand beklagt zu viel Buerokratie und Regulierung",
    "Streaming-Dienste kaempfen um Marktanteile in Deutschland",
    "Logistik-Branche setzt verstaerkt auf autonome Fahrzeuge",
]

# Zusammenfuegen
texts = texts_sport + texts_tech + texts_politik + texts_wirtschaft
labels = (["Sport"] * len(texts_sport) +
          ["Technologie"] * len(texts_tech) +
          ["Politik"] * len(texts_politik) +
          ["Wirtschaft"] * len(texts_wirtschaft))

print(f"Datensatz: {len(texts)} Texte in {len(set(labels))} Kategorien")
for cat in sorted(set(labels)):
    count = labels.count(cat)
    print(f"  {cat}: {count} Texte")

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ============================================================
# 2. Ansatz 1: TF-IDF + Logistic Regression (Baseline)
# ============================================================
print("\n" + "=" * 60)
print("2. ANSATZ 1: TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

start = time.time()
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

clf_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_lr.fit(X_train_tfidf, y_train)
y_pred_lr = clf_lr.predict(X_test_tfidf)
tfidf_time = time.time() - start

acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"\nAccuracy: {acc_lr:.1%}")
print(f"Zeit:     {tfidf_time:.2f}s")
print(f"\n{classification_report(y_test, y_pred_lr)}")

# Top Features pro Kategorie
print("Top-5 Woerter pro Kategorie:")
feature_names = tfidf.get_feature_names_out()
for i, cat in enumerate(clf_lr.classes_):
    top_idx = clf_lr.coef_[i].argsort()[-5:][::-1]
    top_words = [feature_names[j] for j in top_idx]
    print(f"  {cat}: {', '.join(top_words)}")

# ============================================================
# 3. Ansatz 2: Sentence Transformer + Classifier
# ============================================================
print("\n" + "=" * 60)
print("3. ANSATZ 2: SENTENCE TRANSFORMER + CLASSIFIER")
print("=" * 60)

from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingClassifier

start = time.time()
print("Lade multilingual Sentence Transformer...")
st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
X_train_emb = st_model.encode(X_train, show_progress_bar=False)
X_test_emb = st_model.encode(X_test, show_progress_bar=False)

clf_st = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf_st.fit(X_train_emb, y_train)
y_pred_st = clf_st.predict(X_test_emb)
st_time = time.time() - start

acc_st = accuracy_score(y_test, y_pred_st)
print(f"\nAccuracy: {acc_st:.1%}")
print(f"Zeit:     {st_time:.2f}s (inkl. Modell laden)")
print(f"\n{classification_report(y_test, y_pred_st)}")

# ============================================================
# 4. Vergleich
# ============================================================
print("\n" + "=" * 60)
print("4. VERGLEICH DER ANSAETZE")
print("=" * 60)

print(f"\n{'Methode':<35s} {'Accuracy':>10s} {'Zeit':>8s}")
print("-" * 55)
print(f"{'TF-IDF + Logistic Regression':<35s} {acc_lr:>9.1%} {tfidf_time:>7.2f}s")
print(f"{'Sentence Transformer + GBClassifier':<35s} {acc_st:>9.1%} {st_time:>7.2f}s")

print(f"""
Fazit:
  - TF-IDF ist SCHNELL und einfach, oft ueberraschend gut
  - Sentence Transformer versteht Semantik besser (Bedeutung statt nur Woerter)
  - Fuer kleine Datensaetze kann TF-IDF sogar besser sein!
""")

# ============================================================
# 5. Neue Texte klassifizieren
# ============================================================
print("=" * 60)
print("5. NEUE TEXTE KLASSIFIZIEREN")
print("=" * 60)

test_sentences = [
    "Die Mannschaft hat im Elfmeterschiessen verloren",
    "Neues KI-Modell kann Bilder in Sekunden generieren",
    "Bundeskanzler trifft sich mit dem franzoesischen Praesidenten",
    "Aktienkurse fallen nach der Zinsentscheidung der Zentralbank",
    "Schwimmer bricht den Weltrekord ueber 200 Meter Freistil",
    "Hacker stehlen Millionen Datensaetze von Social-Media-Plattform",
]

print()
for text in test_sentences:
    # TF-IDF
    pred_lr = clf_lr.predict(tfidf.transform([text]))[0]
    # Sentence Transformer
    pred_st = clf_st.predict(st_model.encode([text]))[0]

    agree = "[OK]" if pred_lr == pred_st else "[!!]"
    print(f"  {agree} \"{text[:60]}...\"")
    print(f"       TF-IDF: {pred_lr:12s} | Transformer: {pred_st}")

# ============================================================
# 6. Visualisierung
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Text-Klassifikation: TF-IDF vs Sentence Transformer", fontsize=14)

# Confusion Matrix TF-IDF
cm1 = confusion_matrix(y_test, y_pred_lr, labels=sorted(set(labels)))
im1 = axes[0].imshow(cm1, cmap="Blues")
cats = sorted(set(labels))
axes[0].set_xticks(range(len(cats)))
axes[0].set_yticks(range(len(cats)))
axes[0].set_xticklabels(cats, rotation=45, ha="right")
axes[0].set_yticklabels(cats)
axes[0].set_title(f"TF-IDF ({acc_lr:.0%})")
for i in range(len(cats)):
    for j in range(len(cats)):
        axes[0].text(j, i, str(cm1[i, j]), ha="center", va="center",
                     color="white" if cm1[i, j] > cm1.max() / 2 else "black")

# Confusion Matrix Transformer
cm2 = confusion_matrix(y_test, y_pred_st, labels=sorted(set(labels)))
im2 = axes[1].imshow(cm2, cmap="Blues")
axes[1].set_xticks(range(len(cats)))
axes[1].set_yticks(range(len(cats)))
axes[1].set_xticklabels(cats, rotation=45, ha="right")
axes[1].set_yticklabels(cats)
axes[1].set_title(f"Sentence Transformer ({acc_st:.0%})")
for i in range(len(cats)):
    for j in range(len(cats)):
        axes[1].text(j, i, str(cm2[i, j]), ha="center", va="center",
                     color="white" if cm2[i, j] > cm2.max() / 2 else "black")

# Vergleichs-Balkendiagramm
methods = ["TF-IDF + LR", "SentTransformer\n+ GBClassifier"]
accs = [acc_lr, acc_st]
times = [tfidf_time, st_time]
x = np.arange(2)
ax2 = axes[2]
bars = ax2.bar(x, accs, color=["#3498db", "#2ecc71"], width=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy Vergleich")
ax2.set_ylim(0, 1.15)
for bar, acc, t in zip(bars, accs, times):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{acc:.0%}\n({t:.1f}s)", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("text_klassifikation.png", dpi=150)
print("\n[OK] Plot gespeichert als 'text_klassifikation.png'")

# ============================================================
# UEBUNGEN
# ============================================================
print("\n" + "=" * 60)
print("UEBUNGEN")
print("=" * 60)
print("""
1. MEHR KATEGORIEN
   Fuege "Wissenschaft" und "Kultur" als neue Kategorien hinzu
   (je 20+ Beispielsaetze). Wird es schwieriger?

2. ZERO-SHOT CLASSIFICATION
   from transformers import pipeline
   classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
   result = classifier("Text", candidate_labels=["Sport","Tech","Politik","Wirtschaft"])
   Vergleiche mit den trainierten Modellen!

3. FALSCHE VORHERSAGEN ANALYSIEREN
   Finde die Texte, die falsch klassifiziert wurden.
   Warum hat das Modell sich geirrt? Sind die Texte mehrdeutig?

4. EIGENE TEXTE
   Schreibe 10 eigene Saetze und teste beide Modelle.
   Wo sind die Grenzen?
""")
