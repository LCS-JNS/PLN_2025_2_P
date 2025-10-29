import pandas as pd
import numpy as np
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

class AnalisadorNLP:
    """
    Classe para análise comparativa de modelos NLP em múltiplas datas:
    - TF-IDF
    - Word2Vec
    - BERTimbau (BERTugues)
    """
    
    def __init__(self, pasta_base='lista_resultados'):
        """
        Inicializa o analisador carregando dados de múltiplas datas
        
        Args:
            pasta_base: Caminho para a pasta contendo subpastas com datas
        """
        print("="*80)
        print("INICIALIZANDO ANALISADOR NLP - MÚLTIPLAS DATAS")
        print("="*80)
        
        self.pasta_base = pasta_base
        self.dados_por_data = {}
        
        # Carrega dados de todas as datas
        self._carregar_todas_datas()
        
        # Consolida todos os dados
        self._consolidar_dados()
        
        # Inicializa dicionários para armazenar resultados
        self.resultados = {
            'tfidf': {},
            'word2vec': {},
            'bertimbau': {}
        }
        
        print("\n✓ Inicialização concluída!\n")
    
    def _carregar_todas_datas(self):
        """Carrega dados de todas as subpastas (datas)"""
        print(f"\n📂 Buscando dados em: {self.pasta_base}/")
        
        if not os.path.exists(self.pasta_base):
            raise FileNotFoundError(f"Pasta '{self.pasta_base}' não encontrada!")
        
        # Lista todas as subpastas (datas)
        subpastas = [f for f in os.listdir(self.pasta_base) 
                    if os.path.isdir(os.path.join(self.pasta_base, f))]
        
        if not subpastas:
            raise FileNotFoundError(f"Nenhuma subpasta encontrada em '{self.pasta_base}'")
        
        subpastas.sort()  # Ordena por data
        print(f"✓ Encontradas {len(subpastas)} datas: {', '.join(subpastas)}")
        
        # Carrega dados de cada data
        for data in subpastas:
            print(f"\n📅 Processando data: {data}")
            caminho_data = os.path.join(self.pasta_base, data)
            
            try:
                dados = self._carregar_dados_pasta(caminho_data, data)
                self.dados_por_data[data] = dados
                print(f"   ✓ {dados['num_noticias']} notícias carregadas")
            except Exception as e:
                print(f"   ⚠ Erro ao carregar data {data}: {e}")
                continue
        
        if not self.dados_por_data:
            raise ValueError("Nenhum dado foi carregado com sucesso!")
        
        print(f"\n✓ Total de datas processadas: {len(self.dados_por_data)}")
    
    def _carregar_dados_pasta(self, caminho, data):
        """
        Carrega todos os arquivos de uma pasta específica
        
        Args:
            caminho: Caminho da pasta
            data: Nome da data/pasta
            
        Returns:
            dict com todos os dados carregados
        """
        dados = {'data': data}
        
        # 1. Carrega CSV
        csv_files = glob.glob(os.path.join(caminho, 'noticias_g1_processadas*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"CSV não encontrado em {caminho}")
        
        csv_path = csv_files[0]
        print(f"   📄 CSV: {os.path.basename(csv_path)}")
        dados['df'] = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
        dados['num_noticias'] = len(dados['df'])
        
        # 2. Carrega listas de tokens
        arquivos_tokens = {
            'tokens': 'tokens.txt',
            'normalizados': 'normalizados.txt',
            'stems': 'stems.txt',
            'lemmas': 'lemmas.txt'
        }
        
        for nome, arquivo in arquivos_tokens.items():
            caminho_arquivo = os.path.join(caminho, arquivo)
            if os.path.exists(caminho_arquivo):
                dados[nome] = self._carregar_lista(caminho_arquivo)
                print(f"   📝 {arquivo}: {len(dados[nome])} tokens")
            else:
                print(f"   ⚠ {arquivo} não encontrado")
                dados[nome] = []
        
        # 3. Prepara textos completos
        dados['textos'] = (dados['df']['titulo'].fillna('') + ' ' + 
                          dados['df']['subtitulo'].fillna('') + ' ' + 
                          dados['df']['texto'].fillna('')).tolist()
        
        return dados
    
    def _carregar_lista(self, arquivo):
        """Carrega uma lista de tokens de um arquivo texto"""
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                return [linha.strip() for linha in f if linha.strip()]
        except Exception as e:
            print(f"      Erro ao carregar {arquivo}: {e}")
            return []
    
    def _consolidar_dados(self):
        """Consolida dados de todas as datas em estruturas unificadas"""
        print("\n🔄 Consolidando dados de todas as datas...")
        
        # Concatena DataFrames
        dfs = []
        for data, dados in self.dados_por_data.items():
            df_temp = dados['df'].copy()
            df_temp['data_coleta_pasta'] = data
            dfs.append(df_temp)
        
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"   ✓ DataFrame consolidado: {len(self.df)} notícias totais")
        
        # Concatena listas de tokens
        self.tokens_originais = []
        self.tokens_normalizados = []
        self.tokens_stemmed = []
        self.tokens_lemmatizados = []
        
        for data, dados in self.dados_por_data.items():
            self.tokens_originais.extend(dados.get('tokens', []))
            self.tokens_normalizados.extend(dados.get('normalizados', []))
            self.tokens_stemmed.extend(dados.get('stems', []))
            self.tokens_lemmatizados.extend(dados.get('lemmas', []))
        
        print(f"   ✓ Tokens consolidados:")
        print(f"      - Originais: {len(self.tokens_originais)}")
        print(f"      - Normalizados: {len(self.tokens_normalizados)}")
        print(f"      - Stems: {len(self.tokens_stemmed)}")
        print(f"      - Lemmas: {len(self.tokens_lemmatizados)}")
        
        # Concatena textos completos
        self.textos_completos = []
        for data, dados in self.dados_por_data.items():
            self.textos_completos.extend(dados.get('textos', []))
        
        print(f"   ✓ Textos completos: {len(self.textos_completos)}")
        
        # Estatísticas por data
        print("\n📊 Distribuição por data:")
        for data in sorted(self.dados_por_data.keys()):
            num = self.dados_por_data[data]['num_noticias']
            print(f"   • {data}: {num} notícias")
    
    def gerar_tfidf(self, usar_stems=True):
        """
        Gera representação TF-IDF dos documentos consolidados
        
        Args:
            usar_stems: Se True, usa stems; se False, usa lemmas
        """
        print("\n" + "="*80)
        print("1. GERANDO TF-IDF (DADOS CONSOLIDADOS)")
        print("="*80)
        
        tipo = 'stems' if usar_stems else 'lemmas'
        print(f"\n📊 Usando {tipo} para TF-IDF...")
        
        # Prepara documentos: reconstrói textos a partir dos tokens processados
        documentos = []
        coluna = 'tokens_stemmed' if usar_stems else 'tokens_lemmatizados'
        
        for tokens_str in self.df[coluna]:
            try:
                tokens = eval(tokens_str) if isinstance(tokens_str, str) else tokens_str
                if tokens:
                    documentos.append(' '.join(tokens))
                else:
                    documentos.append('')
            except:
                documentos.append('')
        
        print(f"   ✓ {len(documentos)} documentos preparados")
        
        # Cria vetorizador TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=500,  # Top 500 termos
            min_df=2,          # Termo deve aparecer em pelo menos 2 documentos
            max_df=0.8,        # Termo não pode aparecer em mais de 80% dos docs
            ngram_range=(1, 2) # Unigramas e bigramas
        )
        
        # Gera matriz TF-IDF
        tfidf_matrix = vectorizer.fit_transform(documentos)
        feature_names = vectorizer.get_feature_names_out()
        
        print(f"✓ Matriz TF-IDF gerada: {tfidf_matrix.shape}")
        print(f"✓ Vocabulário: {len(feature_names)} termos")
        
        # Armazena resultados
        self.resultados['tfidf'] = {
            'matriz': tfidf_matrix,
            'vectorizer': vectorizer,
            'feature_names': feature_names,
            'tipo': tipo
        }
        
        # Mostra top termos por documento (exemplos de diferentes datas)
        print("\n📌 Top 5 termos TF-IDF por documento (amostra de diferentes datas):")
        datas_unicas = self.df['data_coleta_pasta'].unique()[:3]
        for data in datas_unicas:
            idx = self.df[self.df['data_coleta_pasta'] == data].index[0]
            doc_vector = tfidf_matrix[idx].toarray()[0]
            top_indices = doc_vector.argsort()[-5:][::-1]
            top_terms = [(feature_names[i], doc_vector[i]) for i in top_indices]
            print(f"\n  [{data}] {self.df['titulo'].iloc[idx][:60]}...")
            for term, score in top_terms:
                print(f"    • {term}: {score:.4f}")
        
        return tfidf_matrix
    
    def gerar_word2vec(self, vector_size=100, window=5, min_count=2):
        """
        Treina modelo Word2Vec e gera embeddings dos documentos consolidados
        
        Args:
            vector_size: Dimensão dos vetores
            window: Janela de contexto
            min_count: Frequência mínima de palavras
        """
        print("\n" + "="*80)
        print("2. GERANDO WORD2VEC (DADOS CONSOLIDADOS)")
        print("="*80)
        
        print(f"\n🔧 Parâmetros: dim={vector_size}, window={window}, min_count={min_count}")
        
        # Prepara sentenças tokenizadas para treinar Word2Vec
        sentencas = []
        for tokens_str in self.df['tokens_lemmatizados']:
            try:
                tokens = eval(tokens_str) if isinstance(tokens_str, str) else tokens_str
                if tokens:
                    sentencas.append(tokens)
            except:
                continue
        
        print(f"📝 Treinando Word2Vec com {len(sentencas)} documentos de todas as datas...")
        
        # Treina modelo Word2Vec
        model_w2v = Word2Vec(
            sentences=sentencas,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1,  # Skip-gram
            epochs=10
        )
        
        print(f"✓ Modelo treinado!")
        print(f"✓ Vocabulário: {len(model_w2v.wv)} palavras")
        
        # Gera embeddings dos documentos (média dos vetores das palavras)
        doc_embeddings = []
        for tokens_str in self.df['tokens_lemmatizados']:
            try:
                tokens = eval(tokens_str) if isinstance(tokens_str, str) else tokens_str
                vectors = [model_w2v.wv[word] for word in tokens if word in model_w2v.wv]
                if vectors:
                    doc_embeddings.append(np.mean(vectors, axis=0))
                else:
                    doc_embeddings.append(np.zeros(vector_size))
            except:
                doc_embeddings.append(np.zeros(vector_size))
        
        doc_embeddings = np.array(doc_embeddings)
        
        print(f"✓ Embeddings gerados: {doc_embeddings.shape}")
        
        # Armazena resultados
        self.resultados['word2vec'] = {
            'model': model_w2v,
            'embeddings': doc_embeddings,
            'vocab_size': len(model_w2v.wv)
        }
        
        # Mostra palavras similares (exemplo)
        print("\n📌 Palavras similares no corpus consolidado:")
        palavras_exemplo = ['brasil', 'governo', 'presidente', 'cidade', 'caso']
        for palavra in palavras_exemplo:
            if palavra in model_w2v.wv:
                similares = model_w2v.wv.most_similar(palavra, topn=3)
                print(f"  {palavra}: {', '.join([w for w, s in similares])}")
        
        return doc_embeddings
    
    def gerar_bertimbau(self, model_name='neuralmind/bert-base-portuguese-cased', max_length=512):
        """
        Gera embeddings usando BERTimbau (BERT português) para todos os documentos
        
        Args:
            model_name: Nome do modelo no HuggingFace
            max_length: Comprimento máximo do texto
        """
        print("\n" + "="*80)
        print("3. GERANDO EMBEDDINGS BERTIMBAU (DADOS CONSOLIDADOS)")
        print("="*80)
        
        print(f"\n🤖 Carregando modelo: {model_name}")
        
        # Carrega tokenizer e modelo
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Move para GPU se disponível
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print(f"✓ Modelo carregado em: {device}")
        print(f"\n📝 Gerando embeddings para {len(self.textos_completos)} documentos de todas as datas...")
        
        doc_embeddings = []
        
        # Processa em batches
        batch_size = 8
        total_batches = (len(self.textos_completos) + batch_size - 1) // batch_size
        
        for i in range(0, len(self.textos_completos), batch_size):
            batch_texts = self.textos_completos[i:i+batch_size]
            
            # Tokeniza
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Gera embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Usa [CLS] token embedding (primeiro token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                doc_embeddings.extend(embeddings)
            
            batch_num = i // batch_size + 1
            if batch_num % 10 == 0 or batch_num == total_batches:
                print(f"  Processados: {min(i+batch_size, len(self.textos_completos))}/{len(self.textos_completos)} " +
                      f"({batch_num}/{total_batches} batches)")
        
        doc_embeddings = np.array(doc_embeddings)
        
        print(f"✓ Embeddings gerados: {doc_embeddings.shape}")
        
        # Armazena resultados
        self.resultados['bertimbau'] = {
            'model': model,
            'tokenizer': tokenizer,
            'embeddings': doc_embeddings
        }
        
        return doc_embeddings
    
    def calcular_similaridade_nxn(self):
        """
        Calcula matriz de similaridade N x N para cada modelo
        """
        print("\n" + "="*80)
        print("4. CALCULANDO SIMILARIDADE N x N")
        print("="*80)
        
        for modelo in ['tfidf', 'word2vec', 'bertimbau']:
            if modelo not in self.resultados or not self.resultados[modelo]:
                continue
            
            print(f"\n📊 Similaridade {modelo.upper()}...")
            
            if modelo == 'tfidf':
                matriz = self.resultados['tfidf']['matriz'].toarray()
            else:
                matriz = self.resultados[modelo]['embeddings']
            
            # Calcula similaridade cosseno
            sim_matrix = cosine_similarity(matriz)
            
            # Armazena
            self.resultados[modelo]['similaridade_nxn'] = sim_matrix
            
            # Estatísticas
            triu_indices = np.triu_indices_from(sim_matrix, k=1)
            valores_similares = sim_matrix[triu_indices]
            
            print(f"  ✓ Matriz: {sim_matrix.shape}")
            print(f"  • Similaridade média: {valores_similares.mean():.4f}")
            print(f"  • Desvio padrão: {valores_similares.std():.4f}")
            print(f"  • Mín: {valores_similares.min():.4f}, Máx: {valores_similares.max():.4f}")
            
            # Encontra pares mais similares
            flat_indices = valores_similares.argsort()[-3:][::-1]
            print(f"\n  📌 Top 3 pares mais similares:")
            for idx in flat_indices:
                i, j = triu_indices[0][idx], triu_indices[1][idx]
                sim = sim_matrix[i, j]
                data_i = self.df['data_coleta_pasta'].iloc[i]
                data_j = self.df['data_coleta_pasta'].iloc[j]
                print(f"    • Doc {i+1} ({data_i}) ↔ Doc {j+1} ({data_j}): {sim:.4f}")
                print(f"      - {self.df['titulo'].iloc[i][:50]}...")
                print(f"      - {self.df['titulo'].iloc[j][:50]}...")
    
    def clusterizar_e_visualizar(self, n_clusters=5):
        """
        Realiza clusterização e visualização para cada modelo
        
        Args:
            n_clusters: Número de clusters
        """
        print("\n" + "="*80)
        print("5. CLUSTERIZAÇÃO E VISUALIZAÇÃO")
        print("="*80)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Clusterização com {n_clusters} Clusters - Dados Consolidados ({len(self.df)} documentos)', 
                    fontsize=16, fontweight='bold')
        
        for idx, modelo in enumerate(['tfidf', 'word2vec', 'bertimbau']):
            if modelo not in self.resultados or not self.resultados[modelo]:
                continue
            
            print(f"\n🔍 Clusterizando {modelo.upper()}...")
            
            # Obtém embeddings
            if modelo == 'tfidf':
                X = self.resultados['tfidf']['matriz'].toarray()
            else:
                X = self.resultados[modelo]['embeddings']
            
            # K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            
            # Armazena clusters
            self.resultados[modelo]['clusters'] = clusters
            self.resultados[modelo]['kmeans'] = kmeans
            self.df[f'cluster_{modelo}'] = clusters
            
            # Redução dimensional para visualização (PCA)
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X)
            
            # Visualização
            ax = axes[idx]
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=clusters, cmap='viridis', 
                               alpha=0.6, edgecolors='w', s=100)
            
            # Adiciona centroides
            if modelo == 'tfidf':
                centroides_pca = pca.transform(kmeans.cluster_centers_)
            else:
                centroides_pca = pca.transform(kmeans.cluster_centers_)
            
            ax.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
                      c='red', marker='X', s=300, edgecolors='black', 
                      linewidths=2, label='Centroides')
            
            ax.set_title(f'{modelo.upper()}\nVariância explicada: {pca.explained_variance_ratio_.sum():.2%}',
                        fontweight='bold')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Estatísticas dos clusters
            print(f"  ✓ Inércia: {kmeans.inertia_:.2f}")
            print(f"  • Distribuição dos clusters:")
            unique, counts = np.unique(clusters, return_counts=True)
            for cluster_id, count in zip(unique, counts):
                print(f"    - Cluster {cluster_id}: {count} documentos")
            
            # Mostra exemplos de cada cluster
            print(f"  • Exemplos por cluster:")
            for cluster_id in range(min(3, n_clusters)):  # Mostra apenas 3 primeiros
                docs_cluster = self.df[self.df[f'cluster_{modelo}'] == cluster_id]
                if len(docs_cluster) > 0:
                    exemplo = docs_cluster.iloc[0]
                    print(f"    - Cluster {cluster_id} [{exemplo['data_coleta_pasta']}]: {exemplo['titulo'][:60]}...")
        
        plt.tight_layout()
        plt.savefig('clusters_comparacao_consolidado.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualização salva em: clusters_comparacao_consolidado.png")
        plt.show()
        
        # Dendrograma (para um modelo como exemplo)
        self._gerar_dendrograma('word2vec')
    
    def _gerar_dendrograma(self, modelo='word2vec'):
        """Gera dendrograma para análise hierárquica"""
        print(f"\n📊 Gerando dendrograma para {modelo.upper()}...")
        
        if modelo == 'tfidf':
            X = self.resultados['tfidf']['matriz'].toarray()
        else:
            X = self.resultados[modelo]['embeddings']
        
        # Limita a 50 documentos para legibilidade
        n_samples = min(50, len(X))
        X_sample = X[:n_samples]
        
        # Linkage
        linkage_matrix = linkage(X_sample, method='ward')
        
        # Plot
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, 
                  labels=[f"Doc{i+1}" for i in range(n_samples)],
                  leaf_font_size=8)
        plt.title(f'Dendrograma - {modelo.upper()} (primeiros {n_samples} docs - dados consolidados)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Documentos')
        plt.ylabel('Distância')
        plt.tight_layout()
        plt.savefig(f'dendrograma_{modelo}_consolidado.png', dpi=300, bbox_inches='tight')
        print(f"✓ Dendrograma salvo em: dendrograma_{modelo}_consolidado.png")
        plt.show()
    
    def query_similaridade_1xn(self, query_idx=0, top_n=5):
        """
        Busca documentos similares a uma query (1 x N)
        
        Args:
            query_idx: Índice do documento query
            top_n: Número de documentos similares a retornar
        """
        print("\n" + "="*80)
        print("6. QUERY SIMILARIDADE 1 x N")
        print("="*80)
        
        data_query = self.df['data_coleta_pasta'].iloc[query_idx]
        print(f"\n🔎 Query: Documento {query_idx+1} [{data_query}]")
        print(f"   Título: {self.df['titulo'].iloc[query_idx]}")
        print(f"   Subtítulo: {self.df['subtitulo'].iloc[query_idx][:100]}...")
        
        for modelo in ['tfidf', 'word2vec', 'bertimbau']:
            if modelo not in self.resultados or 'similaridade_nxn' not in self.resultados[modelo]:
                continue
            
            print(f"\n📊 Resultados {modelo.upper()}:")
            
            # Obtém similaridades do documento query
            sim_scores = self.resultados[modelo]['similaridade_nxn'][query_idx]
            
            # Ordena (excluindo o próprio documento)
            similar_indices = sim_scores.argsort()[::-1][1:top_n+1]
            
            print(f"  Top {top_n} documentos mais similares:")
            for rank, idx in enumerate(similar_indices, 1):
                score = sim_scores[idx]
                data_doc = self.df['data_coleta_pasta'].iloc[idx]
                cluster = self.df.get(f'cluster_{modelo}', [None] * len(self.df)).iloc[idx]
                print(f"\n  {rank}. [Score: {score:.4f}] Doc {idx+1} [{data_doc}]")
                print(f"     Título: {self.df['titulo'].iloc[idx][:70]}...")
                print(f"     Cluster: {cluster}")
    
    def comparacao_final(self):
        """
        Gera relatório comparativo final entre os modelos
        """
        print("\n" + "="*80)
        print("7. COMPARAÇÃO FINAL DOS MODELOS")
        print("="*80)
        
        resultados_comparacao = []
        
        for modelo in ['tfidf', 'word2vec', 'bertimbau']:
            if modelo not in self.resultados or not self.resultados[modelo]:
                continue
            
            # Coleta métricas
            if 'similaridade_nxn' in self.resultados[modelo]:
                sim_matrix = self.resultados[modelo]['similaridade_nxn']
                triu_indices = np.triu_indices_from(sim_matrix, k=1)
                similaridades = sim_matrix[triu_indices]
                
                metricas = {
                    'Modelo': modelo.upper(),
                    'Dim. Embeddings': self.resultados[modelo].get('embeddings', 
                                       self.resultados[modelo].get('matriz')).shape[1],
                    'Sim. Média': f"{similaridades.mean():.4f}",
                    'Sim. Std': f"{similaridades.std():.4f}",
                    'Sim. Mín': f"{similaridades.min():.4f}",
                    'Sim. Máx': f"{similaridades.max():.4f}",
                    'Total Docs': len(self.df)
                }
                
                if 'kmeans' in self.resultados[modelo]:
                    metricas['Inércia K-Means'] = f"{self.resultados[modelo]['kmeans'].inertia_:.2f}"
                
                resultados_comparacao.append(metricas)
        
        # Cria DataFrame comparativo
        df_comp = pd.DataFrame(resultados_comparacao)
        
        print("\n📊 TABELA COMPARATIVA (DADOS CONSOLIDADOS):\n")
        print(df_comp.to_string(index=False))
        
        # Visualização comparativa de similaridades
        self._plot_similaridade_comparacao()
        
        # Relatório de cobertura por data
        self._relatorio_cobertura_datas()
    
    def _plot_similaridade_comparacao(self):
        """Plota distribuição de similaridades dos modelos"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Distribuição de Similaridades entre Documentos - Dados Consolidados', 
                    fontsize=16, fontweight='bold')
        
        for idx, modelo in enumerate(['tfidf', 'word2vec', 'bertimbau']):
            if modelo not in self.resultados or 'similaridade_nxn' not in self.resultados[modelo]:
                continue
            
            sim_matrix = self.resultados[modelo]['similaridade_nxn']
            triu_indices = np.triu_indices_from(sim_matrix, k=1)
            similaridades = sim_matrix[triu_indices]
            
            ax = axes[idx]
            ax.hist(similaridades, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(similaridades.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Média: {similaridades.mean():.3f}')
            ax.set_title(f'{modelo.upper()}', fontweight='bold')
            ax.set_xlabel('Similaridade Cosseno')
            ax.set_ylabel('Frequência')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('distribuicao_similaridades_consolidado.png', dpi=300, bbox_inches='tight')
        print("✓ Distribuição salva em: distribuicao_similaridades_consolidado.png")
        plt.show()
    
    def _relatorio_cobertura_datas(self):
        """Gera relatório de cobertura de dados por data"""
        print("\n" + "="*80)
        print("RELATÓRIO DE COBERTURA POR DATA")
        print("="*80)
        
        print(f"\n📊 Estatísticas Gerais:")
        print(f"   • Total de datas processadas: {len(self.dados_por_data)}")
        print(f"   • Total de documentos: {len(self.df)}")
        print(f"   • Período: {min(self.dados_por_data.keys())} a {max(self.dados_por_data.keys())}")
        
        print(f"\n📅 Detalhamento por Data:")
        for data in sorted(self.dados_por_data.keys()):
            num_docs = self.dados_por_data[data]['num_noticias']
            num_tokens = len(self.dados_por_data[data].get('tokens', []))
            num_lemmas = len(self.dados_por_data[data].get('lemmas', []))
            print(f"\n   {data}:")
            print(f"      • Documentos: {num_docs}")
            print(f"      • Tokens: {num_tokens:,}")
            print(f"      • Lemmas: {num_lemmas:,}")
            
            # Mostra distribuição de clusters para esta data (se disponível)
            for modelo in ['tfidf', 'word2vec', 'bertimbau']:
                col_cluster = f'cluster_{modelo}'
                if col_cluster in self.df.columns:
                    docs_data = self.df[self.df['data_coleta_pasta'] == data]
                    dist_clusters = docs_data[col_cluster].value_counts().to_dict()
                    print(f"      • Clusters {modelo}: {dist_clusters}")
    
    def buscar_por_data(self, data, top_n=5):
        """
        Busca documentos dentro de uma data específica
        
        Args:
            data: Data/pasta para buscar
            top_n: Número de documentos a mostrar
        """
        print("\n" + "="*80)
        print(f"BUSCA POR DATA: {data}")
        print("="*80)
        
        if data not in self.dados_por_data:
            print(f"⚠ Data '{data}' não encontrada!")
            print(f"Datas disponíveis: {', '.join(sorted(self.dados_por_data.keys()))}")
            return
        
        docs_data = self.df[self.df['data_coleta_pasta'] == data]
        
        print(f"\n📰 Encontrados {len(docs_data)} documentos em {data}")
        print(f"\n🔝 Top {top_n} notícias:")
        
        for i, (idx, row) in enumerate(docs_data.head(top_n).iterrows(), 1):
            print(f"\n{i}. {row['titulo']}")
            print(f"   Subtítulo: {row['subtitulo'][:80]}...")
            print(f"   Autor: {row.get('autor', 'N/A')}")
            
            # Mostra em qual cluster cada modelo colocou este documento
            for modelo in ['tfidf', 'word2vec', 'bertimbau']:
                col_cluster = f'cluster_{modelo}'
                if col_cluster in self.df.columns:
                    cluster = row[col_cluster]
                    print(f"   Cluster {modelo}: {cluster}")
    
    def comparar_datas(self, data1, data2):
        """
        Compara documentos entre duas datas diferentes
        
        Args:
            data1: Primeira data
            data2: Segunda data
        """
        print("\n" + "="*80)
        print(f"COMPARAÇÃO ENTRE DATAS: {data1} vs {data2}")
        print("="*80)
        
        if data1 not in self.dados_por_data or data2 not in self.dados_por_data:
            print("⚠ Uma ou ambas as datas não foram encontradas!")
            return
        
        docs_data1 = self.df[self.df['data_coleta_pasta'] == data1]
        docs_data2 = self.df[self.df['data_coleta_pasta'] == data2]
        
        print(f"\n📊 Estatísticas:")
        print(f"   {data1}: {len(docs_data1)} documentos")
        print(f"   {data2}: {len(docs_data2)} documentos")
        
        # Análise de similaridade cross-data para cada modelo
        for modelo in ['tfidf', 'word2vec', 'bertimbau']:
            if 'similaridade_nxn' not in self.resultados[modelo]:
                continue
            
            print(f"\n📈 Similaridade Cross-Data - {modelo.upper()}:")
            
            sim_matrix = self.resultados[modelo]['similaridade_nxn']
            
            # Pega índices de cada data
            indices_data1 = docs_data1.index.tolist()
            indices_data2 = docs_data2.index.tolist()
            
            # Calcula similaridades cross-data
            sims_cross = []
            for idx1 in indices_data1:
                for idx2 in indices_data2:
                    sims_cross.append(sim_matrix[idx1, idx2])
            
            if sims_cross:
                print(f"   • Similaridade média entre datas: {np.mean(sims_cross):.4f}")
                print(f"   • Desvio padrão: {np.std(sims_cross):.4f}")
                print(f"   • Máxima similaridade: {np.max(sims_cross):.4f}")
                
                # Encontra o par mais similar entre as datas
                max_sim = 0
                max_pair = (None, None)
                for idx1 in indices_data1:
                    for idx2 in indices_data2:
                        sim = sim_matrix[idx1, idx2]
                        if sim > max_sim:
                            max_sim = sim
                            max_pair = (idx1, idx2)
                
                if max_pair[0] is not None:
                    print(f"\n   📌 Par mais similar entre datas (score: {max_sim:.4f}):")
                    print(f"      [{data1}] {self.df.loc[max_pair[0], 'titulo'][:60]}...")
                    print(f"      [{data2}] {self.df.loc[max_pair[1], 'titulo'][:60]}...")
    
    def executar_analise_completa(self, n_clusters=5, query_idx=0, top_n=5):
        """
        Executa toda a pipeline de análise
        
        Args:
            n_clusters: Número de clusters para K-Means
            query_idx: Índice do documento para query de similaridade
            top_n: Número de documentos similares a retornar
        """
        print("\n" + "="*80)
        print("INICIANDO ANÁLISE COMPLETA - DADOS CONSOLIDADOS")
        print("="*80)
        
        # 1. TF-IDF
        self.gerar_tfidf(usar_stems=True)
        
        # 2. Word2Vec
        self.gerar_word2vec(vector_size=100, window=5, min_count=2)
        
        # 3. BERTimbau
        self.gerar_bertimbau()
        
        # 4. Similaridade N x N
        self.calcular_similaridade_nxn()
        
        # 5. Clusterização e Visualização
        self.clusterizar_e_visualizar(n_clusters=n_clusters)
        
        # 6. Query 1 x N
        self.query_similaridade_1xn(query_idx=query_idx, top_n=top_n)
        
        # 7. Comparação Final
        self.comparacao_final()
        
        print("\n" + "="*80)
        print("✓ ANÁLISE COMPLETA FINALIZADA!")
        print("="*80)
        print("\n📁 Arquivos gerados:")
        print("  • clusters_comparacao_consolidado.png")
        print("  • dendrograma_word2vec_consolidado.png")
        print("  • distribuicao_similaridades_consolidado.png")

def main():
    """
    Função principal para executar a análise
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  ANALISADOR NLP - MÚLTIPLAS DATAS                            ║
    ║  TF-IDF | Word2Vec | BERTimbau                               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Inicializa analisador (carrega dados de lista_resultados/)
    analisador = AnalisadorNLP(pasta_base='lista_resultados')
    
    # Executa análise completa
    analisador.executar_analise_completa(
        n_clusters=5,      # Número de clusters
        query_idx=0,       # Índice do documento para busca de similaridade
        top_n=5           # Top N documentos similares
    )
    
    print("\n" + "="*80)
    print("🎉 ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*80)
    
    print("\n💡 FUNCIONALIDADES ADICIONAIS DISPONÍVEIS:")
    print("\n1. Buscar documentos de uma data específica:")
    print("   analisador.buscar_por_data('2024-10-21', top_n=10)")
    
    print("\n2. Comparar documentos entre duas datas:")
    print("   analisador.comparar_datas('2024-10-21', '2024-10-22')")
    
    print("\n3. Executar query de similaridade personalizada:")
    print("   analisador.query_similaridade_1xn(query_idx=15, top_n=10)")
    
    print("\n4. Gerar modelos individuais:")
    print("   analisador.gerar_tfidf(usar_stems=False)  # Usar lemmas")
    print("   analisador.gerar_word2vec(vector_size=200)")
    print("   analisador.gerar_bertimbau()")
    
    print("\n5. Acessar dados consolidados:")
    print("   df = analisador.df  # DataFrame com todas as notícias")
    print("   datas = analisador.dados_por_data  # Dados separados por data")
    
    print("\n" + "="*80)
    
    # Exemplo de uso adicional (comentado)
    # datas_disponiveis = sorted(analisador.dados_por_data.keys())
    # if len(datas_disponiveis) >= 2:
    #     print(f"\n📊 Exemplo: Comparando primeira e última data...")
    #     analisador.comparar_datas(datas_disponiveis[0], datas_disponiveis[-1])


if __name__ == "__main__":
    main()