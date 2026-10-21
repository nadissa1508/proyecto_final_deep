import {
  CoffeeBeanMascot, IconArrow, IconBars, IconCalendar, IconCloud, IconLeaf,
  MountainsBg,
} from './icons'

const MODULOS = [
  {
    n: '01', titulo: 'Asignador de variedad (CSP)',
    desc: 'Backtracking + Forward Checking + MRV. Recibe altitud, pH y sombra; devuelve la variedad de café compatible con mayor potencial agronómico.',
    iconBg: 'card__icon--sky',
    icon: <IconLeaf />,
  },
  {
    n: '02', titulo: 'Riesgo climático (Bayesian)',
    desc: 'Red bayesiana T, H, P → riesgo de roya → rendimiento esperado. Inferencia por Variable Elimination desde cero, validada contra pgmpy.',
    iconBg: 'card__icon--mint',
    icon: <IconCloud />,
  },
  {
    n: '03', titulo: 'Predictor de cultivo (ML)',
    desc: 'Gaussian Naive Bayes desde cero. Modelo ganador entre KNN, NB, DT, SVM y MLP con F1 macro = 1.0 sobre 22 clases.',
    iconBg: 'card__icon--cream',
    icon: <IconBars />,
  },
  {
    n: '04', titulo: 'Plan estacional (MDP)',
    desc: 'Value Iteration sobre 36 estados (mes × roya) y 5 acciones agronómicas. Política óptima 8.9 % mejor que el mejor baseline constante.',
    iconBg: 'card__icon--coral',
    icon: <IconCalendar />,
  },
]

export default function Landing({ onOpenApp }) {
  return (
    <>
      <section className="section section--sky" style={{ position: 'relative', overflow: 'hidden' }}>
        <div className="section__inner hero">
          <div>
            <div className="hero__brand">
              <span style={{ width: 28, height: 28, borderRadius: '50%', background: 'var(--cm-coffee)', display: 'inline-block' }} />
              CoffeeMind GT
            </div>
            <h1 className="hero__title">
              Cuatro IAs trabajando por tu cafetal.
            </h1>
            <p className="hero__lead">
              Decisiones agronómicas basadas en datos: qué variedad sembrar,
              cuánto riesgo de roya hay, qué cultivo se adapta a tu suelo y
              cómo manejar la temporada mes a mes.
            </p>
            <div className="hero__cta">
              <button className="btn btn--primary" onClick={onOpenApp}>
                Abrir dashboard <IconArrow />
              </button>
              <a href="#modulos" className="btn btn--secondary">Ver los módulos</a>
            </div>
          </div>
          <div className="hero__art">
            <CoffeeBeanMascot size={300} />
          </div>
        </div>
        <MountainsBg />
      </section>

      <section className="section section--neutral" id="modulos">
        <div className="section__inner">
          <h2>Un sistema, cuatro algoritmos clásicos.</h2>
          <p className="muted" style={{ maxWidth: 640, marginTop: '1rem' }}>
            Cada módulo resuelve una pregunta concreta del caficultor.
          </p>

          <div className="modulos">
            {MODULOS.map((m) => (
              <article className="modulo-card" key={m.n}>
                <div className={`card__icon ${m.iconBg}`}>{m.icon}</div>
                <div className="modulo-card__num">Módulo {m.n}</div>
                <div className="modulo-card__title">{m.titulo}</div>
                <p className="modulo-card__desc">{m.desc}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="section section--mint">
        <div className="section__inner">
          <h2>Cómo se usa el dashboard.</h2>
          <p className="muted" style={{ maxWidth: 640, marginTop: '1rem' }}>
            No requiere registro. Tus parcelas viven solo en tu navegador.
          </p>

          <div className="steps">
            <div>
              <div className="step__num">1</div>
              <h3>Agregá una parcela</h3>
              <p className="muted">
                Altitud, pH del suelo, sombra disponible, NPK y condiciones climáticas
                típicas. El dashboard guarda cada parcela en tu navegador.
              </p>
            </div>
            <div>
              <div className="step__num">2</div>
              <h3>Mirá las recomendaciones</h3>
              <p className="muted">
                Variedad sugerida con score, probabilidades de roya y rendimiento,
                cultivo confirmado y plan mensual de tratamientos.
              </p>
            </div>
            <div>
              <div className="step__num">3</div>
              <h3>Comparalo con otras parcelas</h3>
              <p className="muted">
                Cambiá de parcela en la barra lateral; el dashboard recalcula los
                módulos al instante contra el backend en FastAPI.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="section section--sky-deep">
        <div className="section__inner" style={{ textAlign: 'center' }}>
          <h2>Listo para configurar tu primer cafetal.</h2>
          <p style={{ maxWidth: 540, margin: '1rem auto 2rem', opacity: 0.9 }}>
            Es más sencillo de lo que parece y 
            <span className="mono"> no tardas nada</span>.
          </p>
          <button className="btn btn--primary" onClick={onOpenApp}>
            Entrar al dashboard <IconArrow />
          </button>
        </div>
      </section>

      <section className="section section--teal-deep" style={{ paddingTop: 'var(--space-12)', paddingBottom: 'var(--space-12)' }}>
        <div className="section__inner landing-footer">
          <p>
            Proyecto final: CoffeeMind GT | CC3085 - Inteligencia Artificial | Universidad del Valle de Guatemala
          </p>
          <p style={{ marginTop: '0.5rem' }}>
            Copyright © 2026 Grupo 8. Todos los derechos reservados.
          </p>
        </div>
      </section>
    </>
  )
}
