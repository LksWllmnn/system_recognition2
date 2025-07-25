import logging
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import Counter
from sklearn.pipeline import Pipeline
import re
from typing import Dict, List, Tuple
from base_classifier import BaseClassifier
from event import Event, ClassificationResult
from simple_classifier import SimpleEmbeddingClassifier

logger = logging.getLogger(__name__)

# 1. Bag-of-Words mit Naive Bayes (oft besser als TF-IDF bei kleinen Datasets)
class BagOfWordsNaiveBayesClassifier(BaseClassifier):
    def __init__(self, training_data: Dict[str, str]):
        super().__init__("BagOfWordsNB")
        self.pipeline = None
        self.training_data = training_data

    async def initialize(self) -> bool:
        try:
            texts = list(self.training_data.keys())
            labels = list(self.training_data.values())

            # CountVectorizer statt TfidfVectorizer
            self.pipeline = Pipeline([
                ('bow', CountVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.9,
                    max_features=500,  # Kleiner für Naive Bayes
                    token_pattern=r'\b[a-zA-ZäöüÄÖÜß]+\b',
                    lowercase=True
                )),
                ('clf', MultinomialNB(alpha=0.1))  # Laplace-Glättung für kleine Datasets
            ])
            
            self.pipeline.fit(texts, labels)
            logger.info("✅ Bag-of-Words Naive Bayes bereit")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren: {e}")
            return False

    async def classify(self, event: Event) -> ClassificationResult:
        if not self.pipeline:
            return ClassificationResult(event, categories={'unknown': 1.0}, confidence=0.0, 
                                      processing_time=0.0, classifier_name=self.name)

        pred_proba = self.pipeline.predict_proba([event.message])[0]
        labels = self.pipeline.classes_
        categories = dict(zip(labels, pred_proba))

        return ClassificationResult(
            event=event,
            categories=categories,
            confidence=max(categories.values()),
            processing_time=0.0,
            classifier_name=self.name
        )


# 2. Keyword-Density Classifier (sehr robust für technische Domains)
class KeywordDensityClassifier(BaseClassifier):
    def __init__(self, training_data: Dict[str, str]):
        super().__init__("KeywordDensity")
        self.category_keywords = {}
        self.training_data = training_data

    async def initialize(self) -> bool:
        try:
            # Extrahiere charakteristische Keywords pro Kategorie
            category_texts = {}
            for text, category in self.training_data.items():
                if category not in category_texts:
                    category_texts[category] = []
                category_texts[category].append(text.lower())

            # Finde die wichtigsten Keywords pro Kategorie
            for category, texts in category_texts.items():
                # Alle Wörter in dieser Kategorie sammeln
                all_words = []
                for text in texts:
                    words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text)
                    all_words.extend(words)
                
                # Häufigste Wörter finden
                word_counts = Counter(all_words)
                
                # Filtere technische Keywords (länger als 3 Zeichen)
                # Aber behalte mindestens die häufigsten 3 Keywords pro Kategorie
                tech_keywords = {word: count for word, count in word_counts.items() 
                               if len(word) > 3 and count >= 1}
                
                if not tech_keywords:
                    # Fallback: Alle Wörter >3 Zeichen, auch die mit count=1
                    tech_keywords = {word: count for word, count in word_counts.items() 
                                   if len(word) > 3}
                
                if not tech_keywords:
                    # Notfall-Fallback: Die häufigsten Wörter unabhängig von Länge
                    tech_keywords = dict(word_counts.most_common(10))
                
                self.category_keywords[category] = tech_keywords
                
            logger.info("✅ Keyword Density Klassifikator bereit")
            logger.info(f"Keywords: {self.category_keywords}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren: {e}")
            return False

    async def classify(self, event: Event) -> ClassificationResult:
        text = event.message.lower()
        words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text)
        
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = 0.0
            
            if not keywords:  # Keine Keywords gefunden
                score = 0.1   # Minimaler Score für alle Kategorien
            else:
                total_keyword_count = sum(keywords.values())
                
                for word in words:
                    if word in keywords:
                        # Gewichte basierend auf Keyword-Häufigkeit in Trainingsdaten
                        weight = keywords[word] / total_keyword_count
                        score += weight
                
                # Normalisiere durch Textlänge
                score = score / max(len(words), 1)
                
                # Mindest-Score wenn keine Keywords gefunden
                if score == 0:
                    score = 0.01
            
            category_scores[category] = score
        
        # Normalisiere zu Wahrscheinlichkeiten
        total_score = sum(category_scores.values())
        if total_score > 0:
            categories = {k: v/total_score for k, v in category_scores.items()}
        else:
            # Gleichverteilung wenn keine Patterns gefunden
            categories = {k: 1.0/len(self.category_patterns) for k in self.category_patterns.keys()}

        return ClassificationResult(
            event=event,
            categories=categories,
            confidence=max(categories.values()),
            processing_time=0.0,
            classifier_name=self.name
        )


# 3. N-Gram Pattern Classifier (erfasst Wortfolgen)
class NGramPatternClassifier(BaseClassifier):
    def __init__(self, training_data: Dict[str, str]):
        super().__init__("NGramPattern")
        self.category_patterns = {}
        self.training_data = training_data

    async def initialize(self) -> bool:
        try:
            category_texts = {}
            for text, category in self.training_data.items():
                if category not in category_texts:
                    category_texts[category] = []
                category_texts[category].append(text.lower())

            # Extrahiere charakteristische 2-3 Gramme
            for category, texts in category_texts.items():
                patterns = Counter()
                
                for text in texts:
                    words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text)
                    
                    # 2-Gramme
                    for i in range(len(words) - 1):
                        bigram = f"{words[i]} {words[i+1]}"
                        patterns[bigram] += 1
                    
                    # 3-Gramme
                    for i in range(len(words) - 2):
                        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                        patterns[trigram] += 1
                
                # Behalte nur Patterns die mindestens 2x vorkommen
                # Aber mindestens ein Pattern pro Kategorie
                filtered_patterns = {pattern: count for pattern, count in patterns.items() 
                                   if count >= 2}
                if not filtered_patterns and patterns:
                    # Fallback: Nimm die häufigsten Patterns auch wenn sie nur 1x vorkommen
                    filtered_patterns = dict(patterns.most_common(5))
                self.category_patterns[category] = filtered_patterns
            
            logger.info("✅ N-Gram Pattern Klassifikator bereit")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren: {e}")
            return False

    async def classify(self, event: Event) -> ClassificationResult:
        text = event.message.lower()
        words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text)
        
        category_scores = {}
        
        for category, patterns in self.category_patterns.items():
            score = 0.0
            
            if not patterns:  # Keine Patterns gefunden
                score = 0.1   # Minimaler Score statt 0
            else:
                # Prüfe 2-Gramme
                for i in range(len(words) - 1):
                    bigram = f"{words[i]} {words[i+1]}"
                    if bigram in patterns:
                        score += patterns[bigram] * 2  # Höhere Gewichtung für exakte Matches
                
                # Prüfe 3-Gramme
                for i in range(len(words) - 2):
                    trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                    if trigram in patterns:
                        score += patterns[trigram] * 3  # Noch höhere Gewichtung
                
                # Fallback: Einzelwort-Matching wenn keine N-Gramme gefunden
                if score == 0:
                    for word in words:
                        for pattern in patterns:
                            if word in pattern:
                                score += 0.5
            
            category_scores[category] = score
        
        # Normalisiere zu Wahrscheinlichkeiten
        total_score = sum(category_scores.values())
        if total_score > 0:
            categories = {k: v/total_score for k, v in category_scores.items()}
        else:
            categories = {k: 1.0/len(category_scores) for k in category_scores.keys()}

        return ClassificationResult(
            event=event,
            categories=categories,
            confidence=max(categories.values()),
            processing_time=0.0,
            classifier_name=self.name
        )
    
class OptimizedKeywordDensityClassifier(KeywordDensityClassifier):
    async def initialize(self) -> bool:
        try:
            category_texts = {}
            for text, category in self.training_data.items():
                if category not in category_texts:
                    category_texts[category] = []
                category_texts[category].append(text.lower())

            for category, texts in category_texts.items():
                all_words = []
                for text in texts:
                    words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text)
                    all_words.extend(words)
                
                word_counts = Counter(all_words)
                
                # Aggressive Keyword-Extraktion für deutsche Compound-Words
                tech_keywords = {}
                
                # 1. Exact domain keywords (höchste Priorität)
                domain_keywords = {
                    'seil': ['seil', 'tragseil', 'kabel', 'bruch', 'riss', 'prüfung', 'spannung'],
                    'fahrkabine': ['tür', 'türe', 'kabine', 'taste', 'knopf', 'beleuchtung', 'display'],
                    'aufzugsgetriebe': ['motor', 'getriebe', 'antrieb', 'öl', 'vibration', 'geräusch']
                }
                
                # Füge Domain-Keywords hinzu die im Text vorkommen
                for domain_word in domain_keywords.get(category, []):
                    for word, count in word_counts.items():
                        if domain_word in word or word in domain_word:
                            tech_keywords[word] = count * 3  # 3x Gewichtung für Domain-Keywords
                
                # 2. Compound-Word Detection für deutsche Begriffe
                for word, count in word_counts.items():
                    if len(word) > 6:  # Lange Wörter sind oft Compounds
                        for domain_word in domain_keywords.get(category, []):
                            if domain_word in word:
                                tech_keywords[word] = count * 2  # 2x Gewichtung für Compounds
                
                # 3. Häufige Wörter der Kategorie
                for word, count in word_counts.items():
                    if len(word) > 3 and count >= 1:
                        if word not in tech_keywords:
                            tech_keywords[word] = count
                
                # 4. Fallback: Top-Wörter
                if len(tech_keywords) < 5:
                    for word, count in word_counts.most_common(10):
                        if len(word) > 2:
                            tech_keywords[word] = count
                
                self.category_keywords[category] = tech_keywords
                print(f"Optimized Keywords für {category}: {list(tech_keywords.keys())[:10]}")
                
            logger.info("✅ Optimized Keyword Density bereit")
            return True
        except Exception as e:
            logger.error(f"Fehler: {e}")
            return False


class OptimizedNGramPatternClassifier(NGramPatternClassifier):
    async def initialize(self) -> bool:
        try:
            category_texts = {}
            for text, category in self.training_data.items():
                if category not in category_texts:
                    category_texts[category] = []
                category_texts[category].append(text.lower())

            for category, texts in category_texts.items():
                patterns = Counter()
                
                for text in texts:
                    words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text)
                    
                    # 1-Gramme (Einzelwörter) auch hinzufügen
                    for word in words:
                        if len(word) > 3:
                            patterns[word] += 1
                    
                    # 2-Gramme
                    for i in range(len(words) - 1):
                        bigram = f"{words[i]} {words[i+1]}"
                        patterns[bigram] += 1
                    
                    # 3-Gramme
                    for i in range(len(words) - 2):
                        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                        patterns[trigram] += 1
                
                # Weniger restriktive Filterung
                filtered_patterns = {}
                
                # Nimm alle Patterns die mindestens 1x vorkommen
                for pattern, count in patterns.items():
                    if count >= 1:
                        filtered_patterns[pattern] = count
                
                # Mindestens die Top-10 Patterns pro Kategorie
                if len(filtered_patterns) < 10:
                    filtered_patterns = dict(patterns.most_common(10))
                
                self.category_patterns[category] = filtered_patterns
                print(f"Optimized Patterns für {category}: {len(filtered_patterns)} patterns")
                
            logger.info("✅ Optimized N-Gram Pattern bereit")
            return True
        except Exception as e:
            logger.error(f"Fehler: {e}")
            return False


# 4. Random Forest Classifier (robust gegen Overfitting)
class RandomForestTextClassifier(BaseClassifier):
    def __init__(self, training_data: Dict[str, str]):
        super().__init__("RandomForest")
        self.pipeline = None
        self.training_data = training_data

    async def initialize(self) -> bool:
        try:
            texts = list(self.training_data.keys())
            labels = list(self.training_data.values())

            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.85,
                    max_features=300,  # Weniger Features für Random Forest
                    sublinear_tf=True,
                    token_pattern=r'\b[a-zA-ZäöüÄÖÜß]+\b'
                )),
                ('clf', RandomForestClassifier(
                    n_estimators=50,   # Weniger Bäume bei kleinen Datasets
                    max_depth=5,       # Begrenzte Tiefe gegen Overfitting
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight='balanced',
                    random_state=42
                ))
            ])
            
            self.pipeline.fit(texts, labels)
            logger.info("✅ Random Forest Klassifikator bereit")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren: {e}")
            return False

    async def classify(self, event: Event) -> ClassificationResult:
        if not self.pipeline:
            return ClassificationResult(event, categories={'unknown': 1.0}, confidence=0.0, 
                                      processing_time=0.0, classifier_name=self.name)

        pred_proba = self.pipeline.predict_proba([event.message])[0]
        labels = self.pipeline.classes_
        categories = dict(zip(labels, pred_proba))

        return ClassificationResult(
            event=event,
            categories=categories,
            confidence=max(categories.values()),
            processing_time=0.0,
            classifier_name=self.name
        )


# 5. Ensemble Meta-Classifier (kombiniert alle Ansätze)
class EnsembleMetaClassifier(BaseClassifier):
    def __init__(self, training_data: Dict[str, str], classifiers: List[BaseClassifier] = None):
        super().__init__("EnsembleMeta")
        self.classifiers = classifiers or []
        self.training_data = training_data
        # Gewichte basierend auf Debug-Ergebnissen
        self.weights = {
            'BagOfWordsNB': 0.40,         # Sehr stark: 95% und 52% confidence
            'SimpleEmbedding': 0.25,      # Solide: 30% aber konsistent  
            'EnhancedRuleBased': 0.15,    # Gut bei eindeutigen Fällen: 100% für "tür"
            'OllamaLangChain': 0.10,      # Extern aber gut: 80%
            'VotingEnsemble': 0.10,       # Backup
            'TfidfML': 0.0,               # Inkonsistent: 35% vs 36%
            'RandomForest': 0.0,          # Schwach: 40% vs 39%
            'KeywordDensity': 0.0,        # Broken: findet keine Keywords
            'NGramPattern': 0.0,          # Broken: falsche Matches
            'ZeroShot': 0.0               # Schwach und langsam
        }

    def add_classifier(self, classifier: BaseClassifier):
        self.classifiers.append(classifier)

    async def initialize(self) -> bool:
        success = True
        for classifier in self.classifiers:
            classifier_success = await classifier.initialize()
            if not classifier_success:
                logger.warning(f"Klassifikator {classifier.name} konnte nicht initialisiert werden")
                success = False
        
        logger.info("✅ Ensemble Meta-Klassifikator bereit")
        return success

    async def classify(self, event: Event) -> ClassificationResult:
        all_results = {}
        
        # Sammle Ergebnisse von allen Klassifikatoren
        for classifier in self.classifiers:
            try:
                result = await classifier.classify(event)
                all_results[classifier.name] = result.categories
            except Exception as e:
                logger.warning(f"Fehler in Klassifikator {classifier.name}: {e}")
                continue
        
        if not all_results:
            return ClassificationResult(event, categories={'unknown': 1.0}, confidence=0.0, 
                                      processing_time=0.0, classifier_name=self.name)
        
        # Gewichtete Kombination
        combined_categories = {}
        total_weight = 0.0
        
        # Finde alle verfügbaren Kategorien
        all_categories = set()
        for categories in all_results.values():
            all_categories.update(categories.keys())
        
        # Initialisiere alle Kategorien mit 0
        for category in all_categories:
            combined_categories[category] = 0.0
        
        for classifier_name, categories in all_results.items():
            weight = self.weights.get(classifier_name, 0.1)  # Default Gewicht
            total_weight += weight
            
            for category, score in categories.items():
                combined_categories[category] += weight * score
        
        # Normalisiere
        if total_weight > 0:
            combined_categories = {k: v/total_weight for k, v in combined_categories.items()}
        else:
            # Fallback wenn kein Klassifikator funktioniert
            combined_categories = {category: 1.0/len(all_categories) for category in all_categories}
        
        # Confidence-Boost für Konsens
        max_category = max(combined_categories, key=combined_categories.get)
        consensus_boost = self._calculate_consensus_boost(all_results, max_category)
        
        return ClassificationResult(
            event=event,
            categories=combined_categories,
            confidence=max(combined_categories.values()) * (1 + consensus_boost),
            processing_time=0.0,
            classifier_name=self.name
        )
    
    def _calculate_consensus_boost(self, all_results: Dict, predicted_category: str) -> float:
        """Boost wenn mehrere Klassifikatoren sich einig sind"""
        agreements = 0
        total_classifiers = len(all_results)
        
        for categories in all_results.values():
            if predicted_category in categories:
                top_category = max(categories, key=categories.get)
                if top_category == predicted_category:
                    agreements += 1
        
        consensus_ratio = agreements / total_classifiers
        if consensus_ratio >= 0.8:
            return 0.2  # 20% Boost bei hohem Konsens
        elif consensus_ratio >= 0.6:
            return 0.1  # 10% Boost bei mittlerem Konsens
        else:
            return 0.0  # Kein Boost bei niedrigem Konsens


# 6. Voting Classifier (Alternative Ensemble-Methode)
class VotingEnsembleClassifier(BaseClassifier):
    def __init__(self, training_data: Dict[str, str]):
        super().__init__("VotingEnsemble")
        self.classifiers = {}
        self.training_data = training_data

    async def initialize(self) -> bool:
        # Initialisiere verschiedene Klassifikatoren
        self.classifiers['simple'] = SimpleEmbeddingClassifier()
        self.classifiers['nb'] = BagOfWordsNaiveBayesClassifier(self.training_data)
        # Entferne KeywordDensity wegen Attribut-Konflikten
        # self.classifiers['keyword'] = KeywordDensityClassifier(self.training_data)
        
        success = True
        for name, classifier in self.classifiers.items():
            if not await classifier.initialize():
                logger.warning(f"Klassifikator {name} konnte nicht initialisiert werden")
                success = False
        
        logger.info("✅ Voting Ensemble bereit")
        return success

    async def classify(self, event: Event) -> ClassificationResult:
        votes = {}
        confidences = []
        
        for classifier in self.classifiers.values():
            try:
                result = await classifier.classify(event)
                top_category = max(result.categories, key=result.categories.get)
                
                if top_category not in votes:
                    votes[top_category] = 0
                votes[top_category] += 1
                confidences.append(result.confidence)
            except Exception as e:
                logger.warning(f"Fehler in Voting-Klassifikator: {e}")
                continue
        
        if not votes:
            return ClassificationResult(event, categories={'unknown': 1.0}, confidence=0.0, 
                                      processing_time=0.0, classifier_name=self.name)
        
        # Majority Vote
        winner = max(votes, key=votes.get)
        vote_ratio = votes[winner] / sum(votes.values())
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Kategorien basierend auf Votes
        total_votes = sum(votes.values())
        categories = {category: count/total_votes for category, count in votes.items()}
        
        return ClassificationResult(
            event=event,
            categories=categories,
            confidence=vote_ratio * avg_confidence,
            processing_time=0.0,
            classifier_name=self.name
        )

class DebugKeywordDensityClassifier(KeywordDensityClassifier):
    async def classify(self, event: Event) -> ClassificationResult:
        text = event.message.lower()
        words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text)
        
        print(f"\n=== DEBUG KeywordDensity für: '{event.message}' ===")
        print(f"Gefundene Wörter: {words}")
        print(f"Verfügbare Kategorien: {list(self.category_keywords.keys())}")
        
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = 0.0
            matched_keywords = []
            
            if not keywords:
                score = 0.1
                print(f"  {category}: Keine Keywords -> Score: {score}")
            else:
                total_keyword_count = sum(keywords.values())
                
                for word in words:
                    if word in keywords:
                        weight = keywords[word] / total_keyword_count
                        score += weight
                        matched_keywords.append(f"{word}({weight:.3f})")
                
                score = score / max(len(words), 1)
                
                if score == 0:
                    score = 0.01
                
                print(f"  {category}: Keywords={len(keywords)}, Matches={matched_keywords}, Score={score:.4f}")
            
            category_scores[category] = score
        
        # Normalisiere zu Wahrscheinlichkeiten
        total_score = sum(category_scores.values())
        if total_score > 0:
            categories = {k: v/total_score for k, v in category_scores.items()}
        else:
            categories = {k: 1.0/len(category_scores) for k in category_scores.keys()}
        
        print(f"Final Categories: {categories}")
        print("=" * 50)

        return ClassificationResult(
            event=event,
            categories=categories,
            confidence=max(categories.values()),
            processing_time=0.0,
            classifier_name=self.name + "_Debug"
        )


class DebugNGramPatternClassifier(NGramPatternClassifier):
    async def initialize(self) -> bool:
        try:
            category_texts = {}
            for text, category in self.training_data.items():
                if category not in category_texts:
                    category_texts[category] = []
                category_texts[category].append(text.lower())

            print("\n=== DEBUG NGramPattern Initialisierung ===")
            
            for category, texts in category_texts.items():
                patterns = Counter()
                
                print(f"\nKategorie: {category}")
                print(f"Texte: {texts}")
                
                for text in texts:
                    words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text)
                    print(f"  Wörter in '{text}': {words}")
                    
                    # 2-Gramme
                    for i in range(len(words) - 1):
                        bigram = f"{words[i]} {words[i+1]}"
                        patterns[bigram] += 1
                    
                    # 3-Gramme
                    for i in range(len(words) - 2):
                        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                        patterns[trigram] += 1
                
                print(f"  Alle Patterns: {dict(patterns)}")
                
                # Behalte nur Patterns die mindestens 2x vorkommen
                filtered_patterns = {pattern: count for pattern, count in patterns.items() 
                                   if count >= 2}
                if not filtered_patterns and patterns:
                    filtered_patterns = dict(patterns.most_common(5))
                
                self.category_patterns[category] = filtered_patterns
                print(f"  Gefilterte Patterns: {filtered_patterns}")
            
            print("=" * 50)
            logger.info("✅ Debug N-Gram Pattern Klassifikator bereit")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren: {e}")
            return False

    async def classify(self, event: Event) -> ClassificationResult:
        text = event.message.lower()
        words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text)
        
        print(f"\n=== DEBUG NGramPattern für: '{event.message}' ===")
        print(f"Wörter: {words}")
        
        category_scores = {}
        
        for category, patterns in self.category_patterns.items():
            score = 0.0
            matches = []
            
            if not patterns:
                score = 0.1
                print(f"  {category}: Keine Patterns -> Score: {score}")
            else:
                # Prüfe 2-Gramme
                for i in range(len(words) - 1):
                    bigram = f"{words[i]} {words[i+1]}"
                    if bigram in patterns:
                        score += patterns[bigram] * 2
                        matches.append(f"2g:{bigram}({patterns[bigram]})")
                
                # Prüfe 3-Gramme
                for i in range(len(words) - 2):
                    trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                    if trigram in patterns:
                        score += patterns[trigram] * 3
                        matches.append(f"3g:{trigram}({patterns[trigram]})")
                
                # Fallback: Einzelwort-Matching
                if score == 0:
                    for word in words:
                        for pattern in patterns:
                            if word in pattern:
                                score += 0.5
                                matches.append(f"word:{word}->pattern:{pattern}")
                
                print(f"  {category}: Patterns={len(patterns)}, Matches={matches}, Score={score}")
            
            category_scores[category] = score
        
        # Normalisiere zu Wahrscheinlichkeiten
        total_score = sum(category_scores.values())
        if total_score > 0:
            categories = {k: v/total_score for k, v in category_scores.items()}
        else:
            categories = {k: 1.0/len(self.category_patterns) for k in self.category_patterns.keys()}
        
        print(f"Final Categories: {categories}")
        print("=" * 50)

        return ClassificationResult(
            event=event,
            categories=categories,
            confidence=max(categories.values()),
            processing_time=0.0,
            classifier_name=self.name + "_Debug"
        )