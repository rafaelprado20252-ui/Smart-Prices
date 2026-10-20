"""
Smart Price - estimativa de valor de aluguel residencial

App Streamlit com quatro abas: exploracao da base, simulacao de preco,
interpretacao do modelo e informacoes sobre os dados.

Modelo: LightGBM treinado em log(aluguel), R2 0.891, MAPE 21.75%.
Base: 18.564 anuncios do portal ZAP, julho de 2020.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from streamlit_folium import st_folium

plt.rcParams["text.parse_math"] = False

# ----------------------------------------------------------------------
# CONFIGURACAO
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Price",
    layout="wide",
    initial_sidebar_state="expanded",
)

AZUL = "#2563eb"
CINZA = "#6b7280"
CINZA_CLARO = "#e5e7eb"
VERMELHO = "#b91c1c"

COL_RENT = "listing.pricingInfo.rentalPrice"
COL_AREA = "listing.usableAreas"
COL_CITY = "listing.address.city"
COL_NB = "listing.address.neighborhood"
COL_TYPE = "listing.unitTypes"
COL_LAT = "listing.address.point.lat"
COL_LON = "listing.address.point.lon"
COL_PPM2 = "preco_por_m2"

TILE_ESRI = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
    "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)

# rotulos legiveis para as amenidades de lazer
LAZER = {
    "listing.pool": "Piscina",
    "listing.barbgrill": "Churrasqueira",
    "listing.gym": "Academia",
    "listing.sauna": "Sauna",
    "listing.partyhall": "Salao de festas",
    "listing.playground": "Playground",
    "listing.sportcourt": "Quadra poliesportiva",
    "listing.garden": "Jardim",
    "listing.backyard": "Quintal",
    "listing.furnished": "Mobiliado",
    "listing.fireplace": "Lareira",
    "listing.tenniscourt": "Quadra de tenis",
    "listing.bathtub": "Banheira",
    "listing.hottub": "Hidromassagem",
    "listing.mountainview": "Vista para a serra",
    "listing.soundproofing": "Isolamento acustico",
    "listing.guestpark": "Estacionamento de visitantes",
}

# rotulos legiveis para as amenidades de infraestrutura
INFRA = {
    "am_ELEVATOR": "Elevador",
    "am_AIR_CONDITIONING": "Ar-condicionado",
    "am_BALCONY": "Varanda",
    "am_BUILTIN_WARDROBE": "Armario embutido",
    "am_KITCHEN_CABINETS": "Armario de cozinha",
    "am_INTERCOM": "Interfone",
    "am_SERVICE_AREA": "Area de servico",
    "am_GATED_COMMUNITY": "Condominio fechado",
    "am_SECURITY_24_HOURS": "Seguranca 24h",
    "am_WATCHMAN": "Portaria",
    "am_SAFETY_CIRCUIT": "Circuito de seguranca",
    "am_ALARM_SYSTEM": "Alarme",
    "am_CLOSET": "Closet",
    "am_HOME_OFFICE": "Home office",
    "am_GOURMET_SPACE": "Espaco gourmet",
    "am_GOURMET_BALCONY": "Varanda gourmet",
    "am_DINNER_ROOM": "Sala de jantar",
    "am_LAUNDRY": "Lavanderia",
    "am_CABLE_TV": "TV a cabo",
    "am_GARAGE": "Garagem",
    "am_ADULT_GAME_ROOM": "Sala de jogos",
    "am_COPA": "Copa",
    "am_KITCHEN": "Cozinha",
    "am_NUMBER_OF_FLOORS": "Multiplos pavimentos",
}

TIPOS = {
    "APARTMENT": "Apartamento",
    "HOME": "Casa",
    "CONDOMINIUM": "Casa de condominio",
    "FLAT": "Flat",
    "PENTHOUSE": "Cobertura",
    "KITNET": "Kitnet",
    "VILLAGE_HOUSE": "Casa de vila",
}


# ----------------------------------------------------------------------
# CARREGAMENTO
# ----------------------------------------------------------------------

@st.cache_resource
def carrega_modelo():
    return joblib.load("model.joblib")


@st.cache_data
def carrega_dados():
    df = pd.read_parquet("dados_app.parquet")
    return df


@st.cache_resource
def carrega_explainer(_modelo):
    """Construido a partir do modelo em vez de carregado do disco:
    o TreeExplainer e rapido de montar e economiza memoria no deploy."""
    import shap
    return shap.TreeExplainer(_modelo)


@st.cache_data
def resumo_bairros(df):
    """Agregado por cidade e bairro, usado no mapa, no ranking e no
    selo de confianca da simulacao."""
    g = (
        df.groupby([COL_CITY, COL_NB])
        .agg(
            n=(COL_RENT, "size"),
            aluguel=(COL_RENT, "median"),
            ppm2=(COL_PPM2, "median"),
            area=(COL_AREA, "median"),
            lat=(COL_LAT, "median"),
            lon=(COL_LON, "median"),
        )
        .reset_index()
    )
    return g


def estilo_eixo(ax):
    """Grade discreta, sem moldura."""
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    for lado in ["left", "bottom"]:
        ax.spines[lado].set_color(CINZA_CLARO)
    ax.grid(alpha=0.25, color=CINZA_CLARO, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=CINZA, labelsize=9)
    return ax


def brl(v):
    return f"R$ {v:,.0f}".replace(",", ".")


BUNDLE = carrega_modelo()
MODELO = BUNDLE["model"]
FEATURES = BUNDLE["features"]
CAT_FEATURES = BUNDLE["cat_features"]
CAT_CATEGORIES = BUNDLE["cat_categories"]
METRICS = BUNDLE["metrics"]

DF = carrega_dados()
RESUMO = resumo_bairros(DF)

CIDADES = sorted(DF[COL_CITY].dropna().unique())


def monta_entrada(valores: dict) -> pd.DataFrame:
    """Monta uma linha com as colunas na ordem exata do treino e com as
    categorias identicas, que e o que o LightGBM espera."""
    linha = {}
    for f in FEATURES:
        linha[f] = valores.get(f, False if f.startswith("am_") else 0)
    X = pd.DataFrame([linha], columns=FEATURES)

    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c].astype(str), categories=CAT_CATEGORIES[c])

    for f in FEATURES:
        if f not in CAT_FEATURES and X[f].dtype == object:
            X[f] = X[f].astype(float)
    return X


def prevê(valores: dict) -> float:
    X = monta_entrada(valores)
    return float(np.exp(MODELO.predict(X)[0]))


# ----------------------------------------------------------------------
# CABECALHO
# ----------------------------------------------------------------------

st.title("Smart Price")
st.caption(
    "Estimativa de valor de aluguel residencial a partir de caracteristicas "
    "do imovel e da localizacao. Base de 18.564 anuncios, julho de 2020."
)

aba_exp, aba_sim, aba_ent, aba_dados = st.tabs(
    ["Explorar", "Simular", "Entender", "Sobre os dados"]
)


# ----------------------------------------------------------------------
# ABA 1 - EXPLORAR
# ----------------------------------------------------------------------

with aba_exp:
    with st.sidebar:
        st.subheader("Filtros")

        f_cidade = st.selectbox("Cidade", ["Todas"] + list(CIDADES), index=0)

        if f_cidade == "Todas":
            base = DF.copy()
            bairros_disp = []
        else:
            base = DF[DF[COL_CITY] == f_cidade].copy()
            bairros_disp = sorted(base[COL_NB].dropna().unique())

        f_bairros = st.multiselect("Bairro", bairros_disp, default=[])
        if f_bairros:
            base = base[base[COL_NB].isin(f_bairros)]

        tipos_disp = sorted(base[COL_TYPE].dropna().unique())
        f_tipos = st.multiselect(
            "Tipo de imovel",
            tipos_disp,
            default=[],
            format_func=lambda t: TIPOS.get(t, t),
        )
        if f_tipos:
            base = base[base[COL_TYPE].isin(f_tipos)]

        q_max = int(DF["listing.bedrooms"].max())
        f_quartos = st.slider("Quartos", 0, q_max, (0, q_max))
        base = base[base["listing.bedrooms"].between(*f_quartos)]

        f_area = st.slider("Area util (m2)", 10, 1000, (10, 1000))
        base = base[base[COL_AREA].between(*f_area)]

        p_min = int(DF[COL_RENT].min())
        p_max = int(DF[COL_RENT].quantile(0.99))
        f_preco = st.slider("Aluguel (R$)", p_min, p_max, (p_min, p_max), step=100)
        base = base[base[COL_RENT].between(*f_preco)]

    if len(base) == 0:
        st.warning("Nenhum imovel corresponde aos filtros selecionados.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Imoveis", f"{len(base):,}".replace(",", "."))
        c2.metric("Aluguel mediano", brl(base[COL_RENT].median()))
        c3.metric("Preco por m2", brl(base[COL_PPM2].median()))
        c4.metric("Area mediana", f"{base[COL_AREA].median():.0f} m2")

        st.divider()

        col_mapa, col_rank = st.columns([3, 2])

        with col_mapa:
            st.subheader("Distribuicao geografica")
            agg = (
                base.groupby([COL_CITY, COL_NB])
                .agg(
                    n=(COL_RENT, "size"),
                    aluguel=(COL_RENT, "median"),
                    ppm2=(COL_PPM2, "median"),
                    lat=(COL_LAT, "median"),
                    lon=(COL_LON, "median"),
                )
                .reset_index()
                .dropna(subset=["lat", "lon"])
            )
            agg = agg[agg["n"] >= 3]

            if len(agg) == 0:
                st.info("Volume insuficiente para exibir o mapa neste recorte.")
            else:
                m = folium.Map(
                    location=[agg["lat"].median(), agg["lon"].median()],
                    zoom_start=11 if f_cidade != "Todas" else 4,
                    tiles=TILE_ESRI,
                    attr="Esri",
                )
                vmin, vmax = agg["aluguel"].min(), agg["aluguel"].max()
                norm = mcolors.LogNorm(vmin=max(vmin, 1), vmax=max(vmax, vmin + 1))
                cmap = plt.cm.Blues

                for _, r in agg.iterrows():
                    folium.CircleMarker(
                        location=[r["lat"], r["lon"]],
                        radius=4 + np.sqrt(r["n"]),
                        color="#94a3b8",
                        weight=0.6,
                        fill=True,
                        fill_color=mcolors.to_hex(cmap(0.25 + 0.7 * norm(r["aluguel"]))),
                        fill_opacity=0.85,
                        tooltip=f"{r[COL_NB]}: {brl(r['aluguel'])}",
                        popup=folium.Popup(
                            f"<b>{r[COL_NB]}</b><br>"
                            f"{r[COL_CITY]}<br>"
                            f"Anuncios: {int(r['n'])}<br>"
                            f"Aluguel mediano: {brl(r['aluguel'])}<br>"
                            f"Preco por m2: {brl(r['ppm2'])}",
                            max_width=260,
                        ),
                    ).add_to(m)

                st_folium(m, height=460, use_container_width=True,
                          returned_objects=[])
                st.caption(
                    "Tamanho do circulo indica o volume de anuncios; "
                    "a intensidade indica o aluguel mediano."
                )

        with col_rank:
            st.subheader("Ranking de bairros")
            criterio = st.radio(
                "Ordenar por",
                ["Aluguel mediano", "Preco por m2"],
                horizontal=True,
                label_visibility="collapsed",
            )
            col_ord = "aluguel" if criterio == "Aluguel mediano" else "ppm2"

            rank = (
                base.groupby([COL_CITY, COL_NB])
                .agg(n=(COL_RENT, "size"),
                     aluguel=(COL_RENT, "median"),
                     ppm2=(COL_PPM2, "median"))
                .reset_index()
                .query("n >= 5")
                .sort_values(col_ord, ascending=False)
                .head(15)
            )
            if len(rank) == 0:
                st.info("Volume insuficiente para montar o ranking.")
            else:
                fig, ax = plt.subplots(figsize=(5, 6))
                ax.barh(rank[COL_NB][::-1], rank[col_ord][::-1], color=AZUL)
                ax.set_xlabel(criterio, fontsize=9, color=CINZA)
                estilo_eixo(ax)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        st.divider()

        col_hist, col_disp = st.columns(2)

        with col_hist:
            st.subheader("Distribuicao do aluguel")
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.hist(base[COL_RENT], bins=50, color=AZUL, alpha=0.85)
            ax.axvline(base[COL_RENT].median(), color=CINZA, ls="--", lw=1.2)
            ax.set_xlabel("Aluguel (R$)", fontsize=9, color=CINZA)
            ax.set_ylabel("Imoveis", fontsize=9, color=CINZA)
            estilo_eixo(ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_disp:
            st.subheader("Aluguel por area")
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.scatter(base[COL_AREA], base[COL_RENT], s=6, alpha=0.2, color=AZUL)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Area util (m2)", fontsize=9, color=CINZA)
            ax.set_ylabel("Aluguel (R$)", fontsize=9, color=CINZA)
            estilo_eixo(ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with st.expander(f"Ver os {len(base):,} anuncios filtrados".replace(",", ".")):
            cols_tab = [COL_CITY, COL_NB, COL_TYPE, COL_AREA,
                        "listing.bedrooms", "listing.bathrooms",
                        "listing.parkingSpaces", COL_RENT, COL_PPM2]
            cols_tab = [c for c in cols_tab if c in base.columns]
            tab = base[cols_tab].rename(columns={
                COL_CITY: "Cidade", COL_NB: "Bairro", COL_TYPE: "Tipo",
                COL_AREA: "Area (m2)", "listing.bedrooms": "Quartos",
                "listing.bathrooms": "Banheiros",
                "listing.parkingSpaces": "Vagas",
                COL_RENT: "Aluguel", COL_PPM2: "R$/m2",
            })
            st.dataframe(tab.head(500), use_container_width=True, height=360)


# ----------------------------------------------------------------------
# ABA 2 - SIMULAR
# ----------------------------------------------------------------------

with aba_sim:
    st.subheader("Simulacao de valor")

    col_loc, col_car = st.columns(2)

    with col_loc:
        st.markdown("**Localizacao**")
        s_cidade = st.selectbox("Cidade", CIDADES,
                                index=CIDADES.index("São Paulo")
                                if "São Paulo" in CIDADES else 0,
                                key="sim_cidade")
        bairros_cidade = sorted(
            DF.loc[DF[COL_CITY] == s_cidade, COL_NB].dropna().unique()
        )
        s_bairro = st.selectbox("Bairro", bairros_cidade, key="sim_bairro")

        info_nb = RESUMO[(RESUMO[COL_CITY] == s_cidade) &
                         (RESUMO[COL_NB] == s_bairro)]
        n_bairro = int(info_nb["n"].iloc[0]) if len(info_nb) else 0

        if n_bairro >= 30:
            nivel, cor = "Alta", AZUL
        elif n_bairro >= 10:
            nivel, cor = "Media", CINZA
        else:
            nivel, cor = "Baixa", VERMELHO

        st.markdown(
            f"<span style='color:{cor}'>Confianca {nivel.lower()}</span> "
            f"<span style='color:{CINZA}'>&nbsp;|&nbsp; {n_bairro} anuncios "
            f"neste bairro</span>",
            unsafe_allow_html=True,
        )

    with col_car:
        st.markdown("**Caracteristicas**")
        s_tipo = st.selectbox("Tipo de imovel", list(TIPOS.keys()),
                              format_func=lambda t: TIPOS[t], key="sim_tipo")
        s_area = st.slider("Area util (m2)", 10, 1000, 60, key="sim_area")
        cc1, cc2 = st.columns(2)
        s_quartos = cc1.number_input("Quartos", 0, 10, 2, key="sim_q")
        s_banheiros = cc2.number_input("Banheiros", 0, 10, 1, key="sim_b")
        cc3, cc4 = st.columns(2)
        s_suites = cc3.number_input("Suites", 0, 10, 0, key="sim_s")
        s_vagas = cc4.number_input("Vagas", 0, 10, 1, key="sim_v")

    with st.expander("Lazer"):
        cols = st.columns(3)
        sel_lazer = {}
        for i, (col, rot) in enumerate(LAZER.items()):
            sel_lazer[col] = cols[i % 3].checkbox(rot, key=f"lz_{col}")

    with st.expander("Infraestrutura"):
        cols = st.columns(3)
        sel_infra = {}
        for i, (col, rot) in enumerate(INFRA.items()):
            sel_infra[col] = cols[i % 3].checkbox(rot, key=f"if_{col}")

    if len(info_nb):
        lat_nb = float(info_nb["lat"].iloc[0])
        lon_nb = float(info_nb["lon"].iloc[0])
    else:
        lat_nb = float(DF.loc[DF[COL_CITY] == s_cidade, COL_LAT].median())
        lon_nb = float(DF.loc[DF[COL_CITY] == s_cidade, COL_LON].median())

    valores = {
        COL_CITY: s_cidade,
        COL_NB: s_bairro,
        COL_TYPE: s_tipo,
        COL_AREA: float(s_area),
        "log_area": float(np.log(s_area)),
        "listing.bedrooms": float(s_quartos),
        "listing.bathrooms": float(s_banheiros),
        "listing.suites": float(s_suites),
        "listing.parkingSpaces": float(s_vagas),
        COL_LAT: lat_nb,
        COL_LON: lon_nb,
        "n_amenidades": float(sum(sel_lazer.values())),
        "n_tags": float(sum(sel_lazer.values()) + sum(sel_infra.values())),
    }
    valores.update({k: bool(v) for k, v in sel_lazer.items()})
    valores.update({k: bool(v) for k, v in sel_infra.items()})

    st.divider()

    if st.button("Estimar valor", type="primary"):
        estimado = prevê(valores)
        st.session_state["ultima_sim"] = {"valores": valores,
                                          "estimado": estimado,
                                          "bairro": s_bairro,
                                          "cidade": s_cidade,
                                          "area": s_area}

    if "ultima_sim" in st.session_state:
        sim = st.session_state["ultima_sim"]
        est = sim["estimado"]
        margem = METRICS["mape"] / 100

        c1, c2, c3 = st.columns([2, 1, 1])
        c1.metric("Valor estimado", brl(est))
        c2.metric("Faixa inferior", brl(est * (1 - margem)))
        c3.metric("Faixa superior", brl(est * (1 + margem)))

        st.caption(
            f"A faixa corresponde ao erro medio do modelo "
            f"({METRICS['mape']:.1f}%). Em imoveis de alto padrao a "
            f"estimativa tende a ser conservadora."
        )

        if n_bairro < 10:
            st.warning(
                f"O bairro {s_bairro} tem apenas {n_bairro} anuncios na base. "
                "A estimativa e pouco confiavel neste recorte."
            )

        st.divider()
        st.markdown("**Comparacao**")

        comp = DF[(DF[COL_CITY] == sim["cidade"]) & (DF[COL_NB] == sim["bairro"])]
        similares = comp[
            comp[COL_AREA].between(sim["area"] * 0.75, sim["area"] * 1.25)
        ]

        k1, k2, k3 = st.columns(3)
        k1.metric("Mediana do bairro",
                  brl(comp[COL_RENT].median()) if len(comp) else "sem dados")
        k2.metric(f"Similares ({len(similares)} imoveis)",
                  brl(similares[COL_RENT].median()) if len(similares) else "sem dados")
        k3.metric("Preco por m2 estimado", brl(est / sim["area"]))

        if len(comp) >= 5:
            fig, ax = plt.subplots(figsize=(9, 2.6))
            ax.hist(comp[COL_RENT], bins=25, color=CINZA_CLARO)
            ax.axvline(est, color=AZUL, lw=2, label="Estimativa")
            ax.axvline(comp[COL_RENT].median(), color=CINZA, ls="--", lw=1.2,
                       label="Mediana do bairro")
            ax.set_xlabel("Aluguel (R$)", fontsize=9, color=CINZA)
            ax.set_ylabel("Imoveis", fontsize=9, color=CINZA)
            ax.legend(frameon=False, fontsize=8)
            estilo_eixo(ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ----------------------------------------------------------------------
# ABA 3 - ENTENDER
# ----------------------------------------------------------------------

with aba_ent:
    st.subheader("Interpretacao do modelo")

    if "ultima_sim" not in st.session_state:
        st.info("Faca uma simulacao na aba anterior para ver a decomposicao "
                "da estimativa.")
    else:
        sim = st.session_state["ultima_sim"]
        X_sim = monta_entrada(sim["valores"])

        explainer = carrega_explainer(MODELO)
        shap_vals = explainer.shap_values(X_sim)[0]
        base_val = explainer.expected_value

        contrib = (
            pd.DataFrame({"feature": FEATURES, "shap": shap_vals})
            .assign(abs_shap=lambda d: d["shap"].abs())
            .sort_values("abs_shap", ascending=False)
            .head(12)
        )

        st.markdown("**O que mais pesou nesta estimativa**")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        cores = [AZUL if v > 0 else VERMELHO for v in contrib["shap"][::-1]]
        rotulos = [
            LAZER.get(f, INFRA.get(f, f.replace("listing.", "")
                                   .replace("address.", "")))
            for f in contrib["feature"][::-1]
        ]
        ax.barh(rotulos, contrib["shap"][::-1], color=cores)
        ax.axvline(0, color=CINZA, lw=0.8)
        ax.set_xlabel("Impacto no log do aluguel", fontsize=9, color=CINZA)
        estilo_eixo(ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Barras em azul elevam a estimativa; em vermelho, reduzem. "
            "O ponto de partida e o valor medio da base."
        )

        st.divider()
        st.markdown("**Sensibilidade a area**")

        areas = np.linspace(max(20, sim["area"] * 0.4), sim["area"] * 2.2, 25)
        curva = []
        for a in areas:
            v = dict(sim["valores"])
            v[COL_AREA] = float(a)
            v["log_area"] = float(np.log(a))
            curva.append(prevê(v))

        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.plot(areas, curva, color=AZUL, lw=2)
        ax.scatter([sim["area"]], [sim["estimado"]], color=VERMELHO, zorder=5, s=40)
        ax.set_xlabel("Area util (m2)", fontsize=9, color=CINZA)
        ax.set_ylabel("Aluguel estimado (R$)", fontsize=9, color=CINZA)
        estilo_eixo(ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Mantendo todas as demais caracteristicas fixas. O ponto em "
            "vermelho marca a simulacao atual."
        )

    st.divider()
    st.subheader("Importancia global")

    imp = (
        pd.DataFrame({"feature": MODELO.feature_name(),
                      "ganho": MODELO.feature_importance("gain")})
        .sort_values("ganho", ascending=False)
        .head(15)
    )
    imp["pct"] = 100 * imp["ganho"] / imp["ganho"].sum()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    rotulos = [
        LAZER.get(f, INFRA.get(f, f.replace("listing.", "")
                               .replace("address.", "")))
        for f in imp["feature"][::-1]
    ]
    ax.barh(rotulos, imp["pct"][::-1], color=AZUL)
    ax.set_xlabel("Participacao no ganho do modelo (%)", fontsize=9, color=CINZA)
    estilo_eixo(ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    st.subheader("Limitacoes")
    st.markdown(
        """
- A base e de julho de 2020. As estimativas refletem o mercado daquele
  periodo e nao o atual.
- O modelo tende a superestimar imoveis de baixo padrao em cerca de 10%
  e a subestimar imoveis de alto padrao em cerca de 10%. Foram testadas
  tres estrategias de correcao (regressao quantilica, calibracao
  isotonica total e calibracao restrita as caudas) e nenhuma reduziu o
  vies, o que indica limitacao de informacao disponivel e nao de ajuste.
- A cobertura e desigual: 27 cidades concentram 80% dos anuncios. Fora
  delas a estimativa e menos confiavel.
- As coordenadas sao aproximadas. Em parte dos registros correspondem ao
  centroide do bairro, e nao ao endereco exato.
        """
    )


# ----------------------------------------------------------------------
# ABA 4 - SOBRE OS DADOS
# ----------------------------------------------------------------------

with aba_dados:
    st.subheader("Qualidade do modelo")

    c1, c2, c3 = st.columns(3)
    c1.metric("R2", f"{METRICS['r2']:.3f}")
    c2.metric("Erro medio absoluto", brl(METRICS["mae"]))
    c3.metric("Erro percentual medio", f"{METRICS['mape']:.1f}%")

    st.caption(
        "Metricas obtidas por validacao cruzada de 5 folds, calculadas na "
        "escala original em reais."
    )

    st.divider()
    st.subheader("Cobertura por cidade")

    vc = DF[COL_CITY].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(vc.index[::-1], vc.values[::-1], color=AZUL)
    ax.set_xlabel("Numero de anuncios", fontsize=9, color=CINZA)
    estilo_eixo(ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    n_100 = (DF[COL_CITY].value_counts() >= 100).sum()
    st.caption(
        f"A base cobre {DF[COL_CITY].nunique()} cidades. "
        f"{n_100} delas tem ao menos 100 anuncios e concentram a maior "
        f"parte do volume."
    )

    st.divider()
    st.subheader("Metodologia")
    st.markdown(
        """
**Origem.** Anuncios do portal ZAP extraidos em julho de 2020, publicados
no Kaggle. A base bruta tinha 35.772 registros e 78 colunas.

**Tratamento.** Foram mantidos apenas anuncios de aluguel residencial com
periodicidade mensal, o que removeu imoveis a venda e locacao por
temporada. Em seguida foram eliminadas 7.338 duplicatas de anuncio,
valores sentinela em area e custos, e 24 registros cujo preco por metro
quadrado indicava erro de cadastro (preco de venda lancado no campo de
aluguel). A base final tem 18.564 anuncios.

**Variaveis.** Alem das caracteristicas estruturais do imovel, o campo de
amenidades foi decomposto: o texto original continha 156 marcadores
distintos, dos quais apenas 17 estavam disponiveis como colunas. A
extracao das 24 mais frequentes elevou o R2 de 0,886 para 0,889.

**Modelo.** LightGBM treinado sobre o logaritmo do aluguel, com
hiperparametros ajustados por otimizacao bayesiana. Foram comparados sete
modelos sob a mesma validacao cruzada, incluindo XGBoost, CatBoost,
Random Forest e regressao linear regularizada. O ganho sobre a referencia
simples (mediana por cidade e bairro) foi de 50% na reducao do erro
absoluto.

**Principal determinante.** A localizacao responde pela maior parte da
variacao de preco: bairro e cidade juntos explicam 67% da variancia do
logaritmo do aluguel, contra 29% da cidade isolada.
        """
    )
