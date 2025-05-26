BASE_PROMPT = """
Du bist ein Experte darin, Suchanfragen für Google zu formulieren.
Du erhältst einen Textabschnitt.
Formuliere daraus Suchanfragen, die ein Nutzer bei Google eingeben würde, um das Dokument mit dem Text zu finden.

Regeln:
- Schreibe {} Suchanfragen. 
- Variiere die Suchanfragen von einzelnen Suchworten bis zu ganzen Suchphrasen. {} der Suchanfragen sollten nur einzelne Wörter sein.
- Nimm möglichst verschiedene und spezifische Informationen aus dem Textabschnitt auf.
- Gib das Ergebnis als Liste aus.
- Verwende Rechtschreibfehler, um die Suchanfragen realistischer zu gestalten.
- Schreibe {} der Suchanfragen komplett klein, die restlichen mit normalem Casing.
- Verwende Synonyme und verwandte Begriffe.
- Verwende auch Fragen, die ein Nutzer stellen könnte.
- Vermeide es, Jahreszahlen in allen Suchanfragen zu verwenden.

Hier ein Beispiel:
----------------------------------
TEXTABSCHNITT
'Spital Affoltern (Langzeitpflege Sonnenberg, Erweiterung) Das Spital Affoltern verfügt im Bereich der Langzeitversorgung unter dem Namen «Sonnenberg» über insgesamt 128 Betten, verteilt auf die beiden Häuser «Rigi» und «Pilatus», sowie über 17 Betreuungsplätze in einem geriatrischen Tagesheim. Die bestehenden Betten reichen bereits heute nicht zur Deckung des Bedarfs. Die Langzeitpflege Sonnenberg soll deshalb erweitert werden.'

ERGEBNIS
[
    "Spital",
    "affoltern",
    "krankenhaus affoltern",
    "wie heisst die langzeitpfegestation des spitals affoltern",
    "Kapazitätsengpass Spitäler",
    "Statisitk Spitaeler",
    "Hat das Spital Affoltern ein Tagesheim?",
    "geriatrisches tageshiem",
    "langzeitversorgung keine kapazität",
    "Wie viele Betten hat die Langzeitpflege Sonnenberg des Spitals Affoltern?"
  ]

----------------------------------

Hier ist nun dein Textabschnitt, mit dem du arbeiten sollst:
<textabschnitt>
{}
</textabschnitt>
""".strip()
