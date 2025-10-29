import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ClassificadorNoticias:
    """
    Classe para classificação de notícias usando KNN e Naive Bayes
    """
    
    def __init__(self):
        """Inicializa o classificador"""
        print("=" * 80)
        print("CLASSIFICADOR DE NOTÍCIAS - KNN E NAIVE BAYES")
        print("=" * 80)
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.knn_model = None
        self.nb_model = None
        self.resultados_dir = None
        self.knn_dir = None
        self.nb_dir = None
        self.generos_unicos = None
        
        # Configurar diretório de resultados
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.resultados_dir = Path("./resultados_classificacao") / timestamp
        self.resultados_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar subpastas para cada modelo
        self.knn_dir = self.resultados_dir / "KNN"
        self.nb_dir = self.resultados_dir / "NaiveBayes"
        self.knn_dir.mkdir(exist_ok=True)
        self.nb_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 Diretório de resultados: {self.resultados_dir}")
        print(f"  📂 Subpasta KNN: {self.knn_dir}")
        print(f"  📂 Subpasta NaiveBayes: {self.nb_dir}\n")
    
    def carregar_todos_tokens(self):
        """Carrega todos os tokens.txt e CSVs de todas as datas"""
        print("\n" + "=" * 80)
        print("ETAPA 1: CARREGAMENTO DE TODOS OS TOKENS")
        print("=" * 80)
        
        lista_resultados = Path("./lista_resultados")
        
        if not lista_resultados.exists():
            print(f"❌ Diretório não encontrado: {lista_resultados}")
            return False
        
        # Busca todos os CSVs e tokens.txt
        todas_pastas = [p for p in lista_resultados.iterdir() if p.is_dir()]
        
        if not todas_pastas:
            print("❌ Nenhuma pasta encontrada em lista_resultados")
            return False
        
        print(f"📂 Encontradas {len(todas_pastas)} pastas de resultados\n")
        
        todos_tokens = []
        todos_generos = []
        
        for pasta in todas_pastas:
            csv_path = pasta / "noticias_g1_processadas.csv"
            tokens_path = pasta / "tokens.txt"
            
            if not csv_path.exists() or not tokens_path.exists():
                print(f"⚠ Pulando {pasta.name} - arquivos incompletos")
                continue
            
            try:
                # Carrega CSV para pegar gêneros
                df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
                
                # Carrega tokens
                with open(tokens_path, 'r', encoding='utf-8') as f:
                    tokens = [linha.strip() for linha in f if linha.strip()]
                
                # Para cada notícia, agrupa seus tokens
                if 'genero' not in df.columns:
                    print(f"⚠ Pulando {pasta.name} - sem coluna 'genero'")
                    continue
                
                # Remove notícias sem gênero
                df = df[df['genero'].notna() & (df['genero'] != '')]
                
                if len(df) == 0:
                    print(f"⚠ Pulando {pasta.name} - sem notícias válidas")
                    continue
                
                # Calcula quantos tokens por notícia (aproximadamente)
                tokens_por_noticia = len(tokens) // len(df)
                
                # Agrupa tokens por notícia
                idx = 0
                for _, row in df.iterrows():
                    # Pega um bloco de tokens para esta notícia
                    if idx < len(tokens):
                        # Usa total_tokens_originais se disponível
                        n_tokens = row.get('total_tokens_originais', tokens_por_noticia)
                        if pd.isna(n_tokens):
                            n_tokens = tokens_por_noticia
                        n_tokens = int(n_tokens)
                        
                        tokens_noticia = tokens[idx:idx+n_tokens]
                        idx += n_tokens
                        
                        if tokens_noticia:  # Só adiciona se tiver tokens
                            todos_tokens.append(' '.join(tokens_noticia))
                            todos_generos.append(row['genero'])
                
                print(f"✓ {pasta.name}: {len(df)} notícias carregadas")
                
            except Exception as e:
                print(f"❌ Erro ao processar {pasta.name}: {e}")
                continue
        
        if not todos_tokens:
            print("\n❌ Nenhum token carregado!")
            return False
        
        # Cria DataFrame com todos os dados
        self.df = pd.DataFrame({
            'tokens': todos_tokens,
            'genero': todos_generos
        })
        
        print(f"\n✓ Total de documentos carregados: {len(self.df)}")
        print(f"\n📊 Distribuição de gêneros:")
        distribuicao = self.df['genero'].value_counts()
        for genero, count in distribuicao.items():
            print(f"  • {genero}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Remove gêneros com poucas amostras (mínimo 2)
        generos_validos = distribuicao[distribuicao >= 2].index
        self.df = self.df[self.df['genero'].isin(generos_validos)]
        print(f"\n⚠️  Nota: Removendo gêneros com menos de 5 amostras para validação adequada")
        print(f"  • Após filtragem (mín. 5 amostras): {len(self.df)} documentos")
        print(f"  • Gêneros mantidos: {len(generos_validos)}")
        
        # Mostra gêneros removidos
        generos_removidos = set(distribuicao.index) - set(generos_validos)
        if generos_removidos:
            print(f"\n  🗑️  Gêneros removidos (< 5 amostras):")
            for genero in sorted(generos_removidos):
                print(f"    • {genero}: {distribuicao[genero]} amostras")
        
        self.generos_unicos = sorted(generos_validos.tolist())
        print(f"\n  ✅ Gêneros que serão processados ({len(self.generos_unicos)}):")
        for i, genero in enumerate(self.generos_unicos, 1):
            count = distribuicao[genero]
            print(f"    {i:2}. {genero:<35} - {count} amostras")
        
        if len(self.df) < 10:
            print("❌ Dados insuficientes para classificação (mínimo 10 amostras)")
            return False
        
        return True
    
    def preparar_features(self, vectorizer_type='tfidf'):
        """
        Prepara as features para o modelo
        
        Args:
            vectorizer_type: 'tfidf' ou 'count'
        """
        print("\n" + "=" * 80)
        print("ETAPA 2: PREPARAÇÃO DE FEATURES")
        print("=" * 80)
        
        try:
            textos = self.df['tokens'].values
            labels = self.df['genero'].values
            
            print(f"✓ Features preparadas:")
            print(f"  • Tipo de vectorizer: {vectorizer_type.upper()}")
            print(f"  • Total de documentos: {len(textos)}")
            print(f"  • Total de classes: {len(np.unique(labels))}")
            
            # Cria o vectorizer
            if vectorizer_type == 'tfidf':
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8
                )
            else:
                self.vectorizer = CountVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8
                )
            
            # Transforma textos em features
            X = self.vectorizer.fit_transform(textos)
            print(f"  • Dimensão da matriz de features: {X.shape}")
            print(f"  • Features esparsas: {(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.1f}%")
            
            # Split train/test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, labels, test_size=0.25, random_state=42, stratify=labels
            )
            
            print(f"\n✓ Divisão treino/teste:")
            print(f"  • Treino: {self.X_train.shape[0]} amostras")
            print(f"  • Teste: {self.X_test.shape[0]} amostras")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao preparar features: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def treinar_knn(self, otimizar=True):
        """Treina o modelo KNN"""
        print("\n" + "=" * 80)
        print("ETAPA 3: TREINAMENTO KNN")
        print("=" * 80)
        
        try:
            if otimizar:
                print("🔍 Otimizando hiperparâmetros com GridSearch...")
                param_grid = {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'cosine']
                }
                
                knn = KNeighborsClassifier()
                grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                grid_search.fit(self.X_train, self.y_train)
                
                self.knn_model = grid_search.best_estimator_
                print(f"✓ Melhores parâmetros: {grid_search.best_params_}")
                print(f"✓ Melhor score CV: {grid_search.best_score_:.4f}")
            else:
                print("🔨 Treinando KNN com parâmetros padrão...")
                self.knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
                self.knn_model.fit(self.X_train, self.y_train)
            
            # Validação cruzada
            cv_scores = cross_val_score(self.knn_model, self.X_train, self.y_train, cv=5)
            print(f"\n✓ Validação cruzada (5-fold):")
            print(f"  • Scores: {cv_scores}")
            print(f"  • Média: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao treinar KNN: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def treinar_naive_bayes(self, otimizar=True):
        """Treina o modelo Naive Bayes"""
        print("\n" + "=" * 80)
        print("ETAPA 4: TREINAMENTO NAIVE BAYES")
        print("=" * 80)
        
        try:
            if otimizar:
                print("🔍 Otimizando hiperparâmetros com GridSearch...")
                param_grid = {
                    'alpha': [0.1, 0.5, 1.0, 2.0],
                    'fit_prior': [True, False]
                }
                
                nb = MultinomialNB()
                grid_search = GridSearchCV(nb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                grid_search.fit(self.X_train, self.y_train)
                
                self.nb_model = grid_search.best_estimator_
                print(f"✓ Melhores parâmetros: {grid_search.best_params_}")
                print(f"✓ Melhor score CV: {grid_search.best_score_:.4f}")
            else:
                print("🔨 Treinando Naive Bayes com parâmetros padrão...")
                self.nb_model = MultinomialNB(alpha=1.0)
                self.nb_model.fit(self.X_train, self.y_train)
            
            # Validação cruzada
            cv_scores = cross_val_score(self.nb_model, self.X_train, self.y_train, cv=5)
            print(f"\n✓ Validação cruzada (5-fold):")
            print(f"  • Scores: {cv_scores}")
            print(f"  • Média: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao treinar Naive Bayes: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def avaliar_modelo(self, modelo, nome_modelo):
        """Avalia um modelo e retorna métricas"""
        print(f"\n{'=' * 80}")
        print(f"AVALIAÇÃO: {nome_modelo}")
        print("=" * 80)
        
        # Predições
        y_pred_train = modelo.predict(self.X_train)
        y_pred_test = modelo.predict(self.X_test)
        
        # Métricas treino
        acc_train = accuracy_score(self.y_train, y_pred_train)
        
        # Métricas teste
        acc_test = accuracy_score(self.y_test, y_pred_test)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, y_pred_test, average='weighted'
        )
        
        print(f"\n📊 MÉTRICAS GERAIS:")
        print(f"  • Acurácia (Treino): {acc_train:.4f}")
        print(f"  • Acurácia (Teste): {acc_test:.4f}")
        print(f"  • Precisão (Teste): {precision:.4f}")
        print(f"  • Recall (Teste): {recall:.4f}")
        print(f"  • F1-Score (Teste): {f1:.4f}")
        
        # Relatório detalhado por classe
        print(f"\n📋 RELATÓRIO POR CLASSE:")
        print(classification_report(self.y_test, y_pred_test))
        
        return {
            'modelo': nome_modelo,
            'y_pred': y_pred_test,
            'acc_train': acc_train,
            'acc_test': acc_test,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def plotar_matriz_confusao_geral(self, y_true, y_pred, nome_modelo, pasta_destino):
        """Plota e salva matriz de confusão geral - REMOVIDA"""
        pass
    
    def plotar_matriz_confusao_por_genero(self, genero, y_true, y_pred, nome_modelo, pasta_destino):
        """
        Função genérica para plotar matriz de confusão binária de um gênero específico
        
        Args:
            genero: Nome do gênero a ser analisado
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            nome_modelo: Nome do modelo (para título)
            pasta_destino: Path da pasta onde salvar a imagem
        """
        # Converte para binário: genero atual vs outros
        y_true_binary = [genero if y == genero else 'Outros' for y in y_true]
        y_pred_binary = [genero if y == genero else 'Outros' for y in y_pred]
        
        # Matriz de confusão binária
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[genero, 'Outros'])
        
        # Calcula métricas específicas
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[genero, 'Outros'],
            yticklabels=[genero, 'Outros'],
            cbar_kws={'label': 'Quantidade'},
            ax=ax
        )
        
        # Título com métricas
        titulo = f'{nome_modelo}: {genero} vs Outros\n'
        titulo += f'Acc={accuracy:.3f} | Prec={precision:.3f} | Rec={recall:.3f} | F1={f1:.3f}'
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.set_ylabel('Classe Real', fontsize=10)
        ax.set_xlabel('Classe Predita', fontsize=10)
        
        plt.tight_layout()
        
        # Salva com nome limpo (remove caracteres especiais)
        genero_limpo = genero.replace('/', '_').replace(' ', '_').replace('&', 'e')
        arquivo = pasta_destino / f"matriz_{genero_limpo}_vs_outros.png"
        plt.savefig(arquivo, dpi=300, bbox_inches='tight')
        plt.close()
        
        return accuracy, precision, recall, f1
    
    def plotar_matrizes_binarias(self, y_true, y_pred, nome_modelo, pasta_destino):
        """Plota matrizes de confusão binárias para TODOS os gêneros"""
        print(f"\n{'=' * 80}")
        print(f"GERANDO MATRIZES BINÁRIAS - {nome_modelo}")
        print(f"{'=' * 80}")
        print(f"  📊 Total de gêneros a processar: {len(self.generos_unicos)}")
        print(f"  📂 Pasta de destino: {pasta_destino}\n")
        
        metricas_resumo = []
        
        for i, genero in enumerate(self.generos_unicos, 1):
            acc, prec, rec, f1 = self.plotar_matriz_confusao_por_genero(
                genero, y_true, y_pred, nome_modelo, pasta_destino
            )
            print(f"    [{i:2}/{len(self.generos_unicos)}] {genero:<35} - Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
            
            metricas_resumo.append({
                'Gênero': genero,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1
            })
        
        # Salva resumo das métricas por gênero em CSV
        df_resumo = pd.DataFrame(metricas_resumo)
        arquivo_resumo = pasta_destino / "metricas_por_genero.csv"
        df_resumo.to_csv(arquivo_resumo, index=False, encoding='utf-8-sig')
        
        print(f"\n  ✅ {len(self.generos_unicos)} matrizes binárias geradas com sucesso!")
        print(f"  ✅ Resumo de métricas salvo: {arquivo_resumo}")
    
    def comparar_modelos(self, resultados_knn, resultados_nb):
        """Cria visualização comparativa dos modelos"""
        print("\n" + "=" * 80)
        print("COMPARAÇÃO DE MODELOS")
        print("=" * 80)
        
        # Dados para comparação
        metricas = ['Acurácia\n(Treino)', 'Acurácia\n(Teste)', 'Precisão', 'Recall', 'F1-Score']
        knn_valores = [
            resultados_knn['acc_train'],
            resultados_knn['acc_test'],
            resultados_knn['precision'],
            resultados_knn['recall'],
            resultados_knn['f1']
        ]
        nb_valores = [
            resultados_nb['acc_train'],
            resultados_nb['acc_test'],
            resultados_nb['precision'],
            resultados_nb['recall'],
            resultados_nb['f1']
        ]
        
        # Gráfico de barras comparativo
        x = np.arange(len(metricas))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars1 = ax.bar(x - width/2, knn_valores, width, label='KNN', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, nb_valores, width, label='Naive Bayes', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comparação de Performance: KNN vs Naive Bayes', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metricas, fontsize=11)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)
        
        # Adiciona valores nas barras
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9)
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        arquivo = self.resultados_dir / "comparacao_modelos.png"
        plt.savefig(arquivo, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Gráfico comparativo salvo: {arquivo}")
        
        # Tabela comparativa
        print(f"\n📊 TABELA COMPARATIVA:")
        print(f"{'Métrica':<20} {'KNN':>12} {'Naive Bayes':>15} {'Diferença':>12}")
        print("-" * 60)
        for i, metrica in enumerate(['Acurácia (Treino)', 'Acurácia (Teste)', 'Precisão', 'Recall', 'F1-Score']):
            diff = knn_valores[i] - nb_valores[i]
            melhor = "🏆 KNN" if diff > 0 else "🏆 NB" if diff < 0 else "Empate"
            print(f"{metrica:<20} {knn_valores[i]:>12.4f} {nb_valores[i]:>15.4f} {diff:>+11.4f}  {melhor}")
    
    def salvar_metricas_csv(self, resultados_knn, resultados_nb):
        """Salva métricas dos modelos em arquivos CSV"""
        print("\n" + "=" * 80)
        print("SALVANDO MÉTRICAS EM CSV")
        print("=" * 80)
        
        # DataFrame para KNN
        df_knn = pd.DataFrame({
            'Accuracy (Training)': [resultados_knn['acc_train']],
            'Accuracy (Test)': [resultados_knn['acc_test']],
            'Precision': [resultados_knn['precision']],
            'Recall': [resultados_knn['recall']],
            'F1-Score': [resultados_knn['f1']]
        })
        
        arquivo_knn = self.resultados_dir / "KNN_metricas.csv"
        df_knn.to_csv(arquivo_knn, index=False, encoding='utf-8-sig')
        print(f"  ✓ Métricas KNN salvas: {arquivo_knn}")
        
        # DataFrame para Naive Bayes
        df_nb = pd.DataFrame({
            'Accuracy (Training)': [resultados_nb['acc_train']],
            'Accuracy (Test)': [resultados_nb['acc_test']],
            'Precision': [resultados_nb['precision']],
            'Recall': [resultados_nb['recall']],
            'F1-Score': [resultados_nb['f1']]
        })
        
        arquivo_nb = self.resultados_dir / "NaiveBayes_metricas.csv"
        df_nb.to_csv(arquivo_nb, index=False, encoding='utf-8-sig')
        print(f"  ✓ Métricas Naive Bayes salvas: {arquivo_nb}")
    
    def executar_pipeline_completo(self):
        """Executa o pipeline completo de classificação"""
        
        # 1. Carregar todos os tokens
        if not self.carregar_todos_tokens():
            return False
        
        # 2. Preparar features
        if not self.preparar_features(vectorizer_type='tfidf'):
            return False
        
        # 3. Treinar KNN
        if not self.treinar_knn(otimizar=True):
            return False
        
        # 4. Treinar Naive Bayes
        if not self.treinar_naive_bayes(otimizar=True):
            return False
        
        # 5. Avaliar KNN
        print("\n" + "=" * 80)
        print("AVALIAÇÃO E GERAÇÃO DE MATRIZES - KNN")
        print("=" * 80)
        resultados_knn = self.avaliar_modelo(self.knn_model, "KNN")
        self.plotar_matrizes_binarias(self.y_test, resultados_knn['y_pred'], "KNN", self.knn_dir)
        
        # 6. Avaliar Naive Bayes
        print("\n" + "=" * 80)
        print("AVALIAÇÃO E GERAÇÃO DE MATRIZES - NAIVE BAYES")
        print("=" * 80)
        resultados_nb = self.avaliar_modelo(self.nb_model, "NaiveBayes")
        self.plotar_matrizes_binarias(self.y_test, resultados_nb['y_pred'], "NaiveBayes", self.nb_dir)
        
        # 7. Comparar modelos
        self.comparar_modelos(resultados_knn, resultados_nb)
        
        # 8. Salvar métricas em CSV
        self.salvar_metricas_csv(resultados_knn, resultados_nb)
        
        return True


def main():
    """Função principal"""
    
    print("\n" + "=" * 80)
    print("CLASSIFICADOR DE NOTÍCIAS - KNN E NAIVE BAYES")
    print("=" * 80)
    print("\nEste script treina e avalia modelos KNN e Naive Bayes usando")
    print("todos os tokens.txt encontrados em ./lista_resultados\n")
    
    # Cria o classificador e executa
    classificador = ClassificadorNoticias()
    classificador.executar_pipeline_completo()


if __name__ == "__main__":
    main()