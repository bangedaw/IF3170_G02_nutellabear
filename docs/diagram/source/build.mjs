// Generate report diagrams following learning-workflow references/diagram-style.md.
import { writeFileSync } from "node:fs";
import { placeMath } from "./tex-to-svg.mjs";

const OUT = process.argv[2];
const FONT = "Inter, system-ui, sans-serif";
const C = { ink: "#18212f", muted: "#59687d", card: "#dce2e9", plainF: "#f8fafc", plainS: "#bcc7d4", keyF: "#e9f1fc", keyS: "#1d5c9b", refF: "#e9f6f0", refS: "#205c4a" };
const STY = { plain: [C.plainF, C.plainS, 1.5], key: [C.keyF, C.keyS, 2], ref: [C.refF, C.refS, 2] };

const frame = (W, H, title, desc, inner) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-labelledby="t d">
<title id="t">${title}</title>
<desc id="d">${desc}</desc>
<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="${C.muted}"/></marker></defs>
<rect x="1" y="1" width="${W - 2}" height="${H - 2}" rx="12" fill="#ffffff" stroke="${C.card}"/>
${inner.join("\n")}
</svg>
`;
const T = (x, y, s, o = {}) => `<text x="${x}" y="${y}" font-family="${FONT}" font-size="${o.size ?? 14}"${o.bold ? ' font-weight="700"' : ""}${o.ls ? ` letter-spacing="${o.ls}"` : ""} text-anchor="${o.anchor ?? "middle"}" fill="${o.color ?? C.ink}"${o.rot ? ` transform="rotate(${o.rot} ${x} ${y})"` : ""}>${s}</text>`;
const G = (x, y, s, anchor = "start") => T(x, y, s.toUpperCase(), { size: 13, bold: true, ls: ".04em", color: C.muted, anchor });
const box = (x, y, w, h, kind = "plain") => { const [f, s, sw] = STY[kind]; return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${f}" stroke="${s}" stroke-width="${sw}"/>`; };
// node with title line and optional secondary lines (12 px)
const node = (x, y, w, h, kind, title, lines = []) => {
  const n = 1 + lines.length, top = y + h / 2 - (n - 1) * 9 + 5;
  return [box(x, y, w, h, kind), T(x + w / 2, top, title, { bold: true }), ...lines.map((l, i) => T(x + w / 2, top + 18 * (i + 1), l, { size: 12, color: C.muted }))].join("\n");
};
const E = (d, dashed = false, arrow = true, color = C.muted, width = 1.5) => `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}"${dashed ? ' stroke-dasharray="6 4"' : ""}${arrow ? ' marker-end="url(#arr)"' : ""}/>`;
const M = (tex, cx, cy, size = 16) => placeMath(tex, cx, cy, size);
const dot = (x, y, cls) => cls === 0
  ? `<circle cx="${x}" cy="${y}" r="7" fill="${C.keyS}"/>`
  : `<path d="M${x} ${y - 8}L${x + 8} ${y + 6}L${x - 8} ${y + 6}z" fill="${C.refS}"/>`;
const write = (name, svg) => writeFileSync(`${OUT}/${name}.svg`, svg);

// 1. Pipeline
{
  const W = 760, H = 420, I = [];
  I.push(node(20, 40, 150, 56, "ref", "train.csv", ["27.373 baris berlabel"]));
  I.push(node(210, 40, 160, 56, "plain", "Split berstrata", ["80 : 20, seed 42"]));
  I.push(node(410, 18, 150, 48, "plain", "Latih (21.898)"));
  I.push(node(410, 78, 150, 48, "plain", "Validasi (5.475)"));
  I.push(node(596, 18, 144, 56, "plain", "Cleaning", ["duplikat, invalid"]));
  I.push(node(20, 214, 150, 56, "ref", "test.csv", ["6.844 baris"]));
  I.push(E("M170 68H208"), E("M370 60L408 44"), E("M370 76L408 100"), E("M560 42H594"));
  I.push(G(350, 172, "Pipeline: fit pada data latih"));
  I.push(E("M95 214V150H668", false, false), E("M668 74V150", false, false), E("M485 126V150", false, false));
  I.push(E("M335 150V180"), E("M615 150V180"));
  I.push(node(210, 182, 250, 108, "key", "Tampilan numerik", ["fitur turunan, imputasi, clipping", "one-hot, standardisasi", "KNN, LR, SVM"]));
  I.push(node(490, 182, 250, 108, "key", "Tampilan pohon", ["fitur turunan, kategori teks", "nilai hilang dipertahankan", "DTL C4.5, Naive Bayes"]));
  I.push(E("M335 290V318H615V290", false, false), E("M335 318V342"), E("M615 318V342"));
  I.push(node(210, 344, 250, 56, "plain", "Evaluasi validasi", ["akurasi, presisi, recall, F1, ROC-AUC"]));
  I.push(node(490, 344, 250, 56, "ref", "Submission Kaggle", ["fit ulang seluruh train, tulis CSV"]));
  write("pipeline", frame(W, H, "Alur pengolahan data",
    "train.csv dibagi berstrata 80:20 menjadi data latih dan validasi. Data latih dibersihkan, lalu PreparationPipeline di-fit hanya pada data latih bersih. Data validasi dan test.csv hanya di-transform. Pipeline menghasilkan tampilan numerik untuk KNN, regresi logistik, dan SVM, serta tampilan pohon untuk DTL C4.5 dan Naive Bayes. Keduanya mendahului evaluasi validasi dan submission Kaggle.", I));
}

// 2. K-fold
{
  const W = 760, H = 300, I = [];
  I.push(G(20, 32, "Stratified 5-fold cross-validation"));
  for (let r = 0; r < 5; r++) {
    const y = 50 + r * 38;
    I.push(T(70, y + 20, `Lipatan ${r + 1}`, { anchor: "end", size: 13 }));
    for (let c = 0; c < 5; c++) I.push(`<rect x="${90 + c * 112}" y="${y}" width="108" height="30" rx="6" fill="${c === r ? C.keyF : C.plainF}" stroke="${c === r ? C.keyS : C.plainS}" stroke-width="${c === r ? 2 : 1.5}"/>`, T(144 + c * 112, y + 20, c === r ? "validasi" : "latih", { size: 12, color: c === r ? C.ink : C.muted }));
    I.push(E(`M650 ${y + 15}H684`));
  }
  I.push(node(688, 108, 56, 104, "ref", "Skor", ["rata-", "rata"]));
  I.push(`<rect x="90" y="252" width="22" height="16" rx="4" fill="${C.plainF}" stroke="${C.plainS}" stroke-width="1.5"/>`, T(120, 265, "data latih lipatan, pipeline di-fit di sini", { anchor: "start", size: 12, color: C.muted }));
  I.push(`<rect x="440" y="252" width="22" height="16" rx="4" fill="${C.keyF}" stroke="${C.keyS}" stroke-width="2"/>`, T(470, 265, "data validasi lipatan", { anchor: "start", size: 12, color: C.muted }));
  write("kfold", frame(W, H, "Stratified 5-fold cross-validation",
    "Data dibagi menjadi lima lipatan dengan proporsi kelas yang sama. Pada lipatan ke-i, bagian ke-i menjadi data validasi dan empat bagian lain menjadi data latih tempat pipeline di-fit. Lima skor validasi kemudian dirata-rata.", I));
}

// 3. KNN
{
  const W = 760, H = 360, I = [];
  I.push(`<rect x="20" y="20" width="420" height="320" rx="8" fill="${C.plainF}" stroke="${C.plainS}" stroke-width="1.5"/>`);
  I.push(`<circle cx="230" cy="180" r="62" fill="none" stroke="${C.keyS}" stroke-width="2" stroke-dasharray="6 4"/>`);
  I.push(`<circle cx="230" cy="180" r="118" fill="none" stroke="${C.muted}" stroke-width="1.5" stroke-dasharray="3 4"/>`);
  [[270, 150], [190, 210], [90, 80], [70, 270], [120, 310], [335, 265], [150, 130], [318, 112], [100, 180]].forEach(([x, y]) => I.push(dot(x, y, 0)));
  [[245, 127], [268, 215], [193, 157], [395, 62], [405, 215], [378, 150], [300, 44], [370, 312]].forEach(([x, y]) => I.push(dot(x, y, 1)));
  I.push(`<rect x="223" y="173" width="14" height="14" fill="${C.ink}"/>`, M(String.raw`x_q`, 254, 194, 16));
  const L = 470;
  I.push(`<rect x="${L - 8}" y="34" width="14" height="14" fill="${C.ink}"/>`, T(L + 16, 46, "titik kueri", { anchor: "start" }));
  I.push(dot(L, 72, 0), T(L + 16, 77, "kelas 0, tidak batal (lingkaran)", { anchor: "start" }));
  I.push(dot(L, 102, 1), T(L + 16, 107, "kelas 1, batal (segitiga)", { anchor: "start" }));
  I.push(E(`M${L - 10} 136H${L + 22}`, true, false, C.keyS, 2), T(L + 30, 140, "k = 5: tiga batal, dua tidak", { anchor: "start", size: 13 }));
  I.push(M(String.raw`\hat{p}(x_q)=\tfrac{3}{5}=0{,}6`, L + 120, 168, 16));
  I.push(E(`M${L - 10} 206H${L + 22}`, true, false, C.muted, 1.5), T(L + 30, 210, "k lebih besar", { anchor: "start", size: 13 }));
  I.push(T(L + 30, 230, "batas keputusan lebih halus,", { anchor: "start", size: 12, color: C.muted }));
  I.push(T(L + 30, 248, "bias naik, variansi turun", { anchor: "start", size: 12, color: C.muted }));
  I.push(T(L - 10, 286, "Pembobotan jarak:", { anchor: "start", size: 13, bold: true }));
  I.push(M(String.raw`w_i=\frac{1}{d(x_q,x_i)}`, L + 110, 316, 16));
  write("knn", frame(W, H, "Ilustrasi k-Nearest Neighbors",
    "Titik kueri x_q dikelilingi titik kelas 0 (lingkaran) dan kelas 1 (segitiga). Lingkaran putus-putus pertama memuat lima tetangga terdekat: tiga segitiga dan dua lingkaran, sehingga peluang batal sama dengan 3/5 = 0,6. Lingkaran kedua menunjukkan k yang lebih besar. Pada pembobotan jarak, w_i sama dengan 1 dibagi d(x_q, x_i).", I));
}

// 4. C4.5 missing values
{
  const W = 760, H = 340, I = [];
  I.push(box(200, 20, 360, 70, "key"), T(380, 44, "Uji atribut days_before_arrival", { bold: true }));
  I.push(M(String.raw`x\le t\,?`, 380, 62, 16), T(380, 82, "100 sampel: 60 kiri, 30 kanan, 10 hilang", { size: 12, color: C.muted }));
  I.push(E("M300 90L180 150"), E("M460 90L580 150"));
  I.push(M(String.raw`p=\tfrac{60}{90}`, 200, 112, 16), M(String.raw`p=\tfrac{30}{90}`, 560, 112, 16));
  I.push(box(40, 152, 280, 84, "plain"), T(180, 176, "Cabang kiri", { bold: true }), T(180, 196, "60 sampel berbobot 1 ditambah", { size: 12, color: C.muted }), T(222, 220, "10 sampel hilang, bobot", { size: 12, color: C.muted, anchor: "end" }), M(String.raw`\tfrac{2}{3}`, 236, 216, 16));
  I.push(box(440, 152, 280, 84, "plain"), T(580, 176, "Cabang kanan", { bold: true }), T(580, 196, "30 sampel berbobot 1 ditambah", { size: 12, color: C.muted }), T(622, 220, "10 sampel hilang, bobot", { size: 12, color: C.muted, anchor: "end" }), M(String.raw`\tfrac{1}{3}`, 636, 216, 16));
  ["Gain dihitung pada sampel yang nilainya diketahui, lalu dikali proporsi sampel diketahui.",
   "Split information memperlakukan nilai hilang sebagai satu partisi tambahan.",
   "Saat prediksi, nilai hilang menelusuri semua cabang; peluang daun digabung",
   "sesuai bobot cabang masing-masing."].forEach((s, i) => I.push(T(380, 266 + i * 18, s, { size: 12, color: C.muted })));
  write("dtree", frame(W, H, "Bobot pecahan nilai hilang pada C4.5",
    "Simpul menguji apakah days_before_arrival x kurang dari atau sama dengan t. Dari 100 sampel, 60 ke kiri, 30 ke kanan, dan 10 bernilai hilang. Peluang cabang kiri p = 60/90 dan kanan p = 30/90. Sepuluh sampel hilang dikirim ke kedua cabang dengan bobot 2/3 ke kiri dan 1/3 ke kanan.", I));
}

// 5. Logistic regression
{
  const W = 780, H = 340, I = [];
  I.push(G(30, 30, "Fungsi sigmoid"));
  I.push(E("M40 250H350", false, true), E("M195 270V48", false, true));
  I.push(E("M40 70H350", true, false, C.plainS), E("M40 160H350", true, false, C.plainS));
  const pts = []; for (let z = -6; z <= 6.001; z += 0.25) pts.push(`${(195 + z * 25).toFixed(1)} ${(250 - 180 / (1 + Math.exp(-z))).toFixed(1)}`);
  I.push(`<path d="M${pts.join("L")}" fill="none" stroke="${C.keyS}" stroke-width="2.5"/>`);
  I.push(T(32, 74, "1", { anchor: "end", size: 12, color: C.muted }), T(32, 164, "0,5", { anchor: "end", size: 12, color: C.muted }), T(32, 254, "0", { anchor: "end", size: 12, color: C.muted }));
  I.push(M("z", 362, 252, 16), M(String.raw`\sigma(z)`, 222, 46, 16));
  I.push(M(String.raw`\sigma(z)=\frac{1}{1+e^{-z}},\quad z=\mathbf{w}^\top\mathbf{x}+b`, 195, 302, 17));
  const ox = 590, oy = 175;
  I.push(G(430, 30, "Daerah penalti dua bobot"));
  I.push(E(`M${ox - 140} ${oy}H${ox + 140}`), E(`M${ox} ${oy + 120}V${oy - 125}`));
  I.push(`<ellipse cx="${ox + 70}" cy="${oy - 92}" rx="52" ry="24" fill="none" stroke="${C.plainS}" stroke-width="1.5" transform="rotate(-25 ${ox + 70} ${oy - 92})"/>`);
  I.push(`<ellipse cx="${ox + 70}" cy="${oy - 92}" rx="82" ry="40" fill="none" stroke="${C.plainS}" stroke-width="1.5" transform="rotate(-25 ${ox + 70} ${oy - 92})"/>`);
  I.push(`<path d="M${ox} ${oy - 60}L${ox + 60} ${oy}L${ox} ${oy + 60}L${ox - 60} ${oy}z" fill="${C.keyF}" stroke="${C.keyS}" stroke-width="2"/>`);
  I.push(`<circle cx="${ox}" cy="${oy}" r="60" fill="none" stroke="${C.refS}" stroke-width="1.5"/>`);
  I.push(`<path d="M${ox} ${oy - 60}C${ox + 26} ${oy - 58} ${ox + 57} ${oy - 26} ${ox + 60} ${oy}C${ox + 57} ${oy + 26} ${ox + 26} ${oy + 58} ${ox} ${oy + 60}C${ox - 26} ${oy + 58} ${ox - 57} ${oy + 26} ${ox - 60} ${oy}C${ox - 57} ${oy - 26} ${ox - 26} ${oy - 58} ${ox} ${oy - 60}z" fill="none" stroke="${C.muted}" stroke-width="1.5" stroke-dasharray="6 4"/>`);
  I.push(`<circle cx="${ox}" cy="${oy - 60}" r="5" fill="${C.ink}"/>`);
  I.push(M("w_1", ox + 150, oy + 16, 16), M("w_2", ox + 18, oy - 128, 16));
  I.push(E("M440 290H470", false, false, C.keyS, 2), T(478, 294, "L1: belah ketupat, solusi di sudut", { anchor: "start", size: 12 }));
  I.push(E("M440 310H470", false, false, C.refS, 1.5), T(478, 314, "L2: lingkaran, bobot mengecil", { anchor: "start", size: 12 }));
  I.push(E("M440 330H470", true, false, C.muted, 1.5), T(478, 334, "Elastic Net: gabungan keduanya", { anchor: "start", size: 12 }));
  write("logreg", frame(W, H + 10, "Sigmoid dan daerah penalti regularisasi",
    "Kiri: kurva sigmoid sigma(z) = 1/(1 + e^(-z)) dengan z = w transpos x + b, bernilai 0,5 pada z = 0 dan mendekati 0 dan 1 di kedua ujung. Kanan: pada bidang dua bobot w1 dan w2, daerah penalti L1 berbentuk belah ketupat, L2 berbentuk lingkaran, dan Elastic Net berada di antaranya. Kontur loss pertama kali menyentuh daerah L1 di sudut pada sumbu w2, sehingga w1 bernilai tepat nol.", I));
}

// 6. SVM
{
  const W = 790, H = 362, I = [];
  I.push(`<rect x="20" y="20" width="360" height="300" rx="8" fill="${C.plainF}" stroke="${C.plainS}" stroke-width="1.5"/>`);
  I.push(`<path d="M60 310L340 30" fill="none" stroke="${C.ink}" stroke-width="2"/>`);
  I.push(E("M20 271L270 21", true, false), E("M130 320L380 70", true, false));
  [[70, 110], [110, 70], [60, 180], [150, 60], [100, 150], [140, 151], [198, 92]].forEach(([x, y]) => I.push(dot(x, y, 0)));
  [[322, 240], [340, 200], [285, 300], [345, 290], [252, 199], [355, 150]].forEach(([x, y]) => I.push(dot(x, y, 1)));
  [[140, 151], [252, 199], [198, 92]].forEach(([x, y]) => I.push(`<circle cx="${x}" cy="${y}" r="13" fill="none" stroke="${C.ink}" stroke-width="1.5"/>`));
  I.push(`<path d="M172 188L213 229" fill="none" stroke="${C.muted}" stroke-width="1.5" marker-start="url(#arr)" marker-end="url(#arr)"/>`);
  I.push(T(224, 246, "margin", { size: 12, color: C.muted, anchor: "start" }), M(String.raw`\frac{2}{\lVert\mathbf{w}\rVert}`, 244, 270, 16));
  I.push(T(30, 344, "garis utuh:", { anchor: "start", size: 12, color: C.muted }), M(String.raw`\mathbf{w}^\top\mathbf{x}+b=0`, 150, 340, 16));
  I.push(T(230, 344, "lingkaran hitam: support vector", { anchor: "start", size: 12, color: C.muted }));
  const X = 410;
  I.push(G(X, 34, "Kernel trick"));
  I.push(`<rect x="${X}" y="50" width="150" height="110" rx="8" fill="${C.plainF}" stroke="${C.plainS}" stroke-width="1.5"/>`);
  [[X + 65, 100], [X + 85, 92], [X + 78, 115]].forEach(([x, y]) => I.push(dot(x, y, 1)));
  [[X + 25, 70], [X + 130, 75], [X + 20, 140], [X + 125, 145], [X + 75, 62], [X + 75, 150]].forEach(([x, y]) => I.push(dot(x, y, 0)));
  I.push(T(X + 75, 180, "ruang asal", { size: 12, color: C.muted }));
  I.push(E(`M${X + 160} 105H${X + 205}`), M(String.raw`\phi(\mathbf{x})`, X + 182, 88, 16));
  I.push(`<rect x="${X + 210}" y="50" width="150" height="110" rx="8" fill="${C.plainF}" stroke="${C.plainS}" stroke-width="1.5"/>`);
  [[X + 240, 140], [X + 262, 145], [X + 284, 138]].forEach(([x, y]) => I.push(dot(x, y, 1)));
  [[X + 232, 70], [X + 262, 64], [X + 300, 74], [X + 330, 66], [X + 342, 84], [X + 316, 94]].forEach(([x, y]) => I.push(dot(x, y, 0)));
  I.push(`<path d="M${X + 215} 115L${X + 355} 110" fill="none" stroke="${C.ink}" stroke-width="2"/>`);
  I.push(T(X + 285, 180, "ruang fitur", { size: 12, color: C.muted }));
  I.push(T(X, 214, "Kernel menghitung hasil kali dalam di ruang fitur:", { anchor: "start", size: 13 }));
  I.push(T(X, 246, "linear", { anchor: "start", size: 13, bold: true }), M(String.raw`K=\mathbf{x}^\top\mathbf{x}'`, X + 230, 242, 16));
  I.push(T(X, 280, "polinomial", { anchor: "start", size: 13, bold: true }), M(String.raw`K=(\gamma\,\mathbf{x}^\top\mathbf{x}'+r)^d`, X + 230, 276, 16));
  I.push(T(X, 314, "Gaussian (RBF)", { anchor: "start", size: 13, bold: true }), M(String.raw`K=\exp\!\left(-\gamma\lVert\mathbf{x}-\mathbf{x}'\rVert^2\right)`, X + 230, 310, 16));
  write("svm", frame(W, H, "Margin SVM dan kernel trick",
    "Kiri: hyperplane w transpos x + b = 0 memisahkan kelas 0 (lingkaran) dan kelas 1 (segitiga). Dua garis putus-putus adalah batas margin dengan lebar 2 dibagi norma w. Tiga titik berlingkar hitam pada garis margin adalah support vector. Kanan: pemetaan phi(x) membuat kelas yang melingkar di ruang asal menjadi terpisah garis di ruang fitur. Kernel linear K = x transpos x', polinomial K = (gamma x transpos x' + r) pangkat d, dan Gaussian K = exp(-gamma kuadrat norma x - x').", I));
}

// 7. Naive Bayes
{
  const W = 760, H = 300, I = [];
  I.push(box(300, 22, 160, 50, "key"), T(380, 44, "Kelas", { bold: true }), M(String.raw`C`, 380, 60, 16));
  const xs = [90, 245, 400, 650];
  ["X_1", "X_2", "X_3", "X_n"].forEach((tex, i) => { I.push(box(xs[i] - 60, 150, 120, 48, "plain"), M(tex, xs[i], 174, 17)); I.push(E(`M${380 + (i - 1.5) * 30} 72L${xs[i]} 148`)); });
  I.push(T(525, 180, "...", { size: 14, color: C.muted }));
  I.push(M(String.raw`P(C\mid X_1,\dots,X_n)\propto P(C)\prod_{i=1}^{n}P(X_i\mid C)`, 380, 236, 17));
  I.push(T(380, 280, "tidak ada busur antarfitur: fitur saling bebas bila kelas diketahui", { size: 12, color: C.muted }));
  write("naive_bayes", frame(W, H, "Struktur Naive Bayes",
    "Simpul kelas C menjadi satu-satunya induk bagi setiap fitur X1 sampai Xn, dan tidak ada busur antarfitur. Peluang posterior P(C | X1, ..., Xn) sebanding dengan P(C) dikali hasil kali P(Xi | C) untuk i = 1 sampai n.", I));
}

// 8. Confusion matrix
{
  const W = 580, H = 300, I = [];
  I.push(T(330, 30, "Prediksi", { bold: true }), T(255, 54, "0, tidak batal", { size: 12, color: C.muted }), T(405, 54, "1, batal", { size: 12, color: C.muted }));
  I.push(T(46, 160, "Aktual", { bold: true, rot: -90 }), T(140, 112, "0", { size: 12, color: C.muted }), T(140, 202, "1", { size: 12, color: C.muted }));
  const cell = (x, y, kind, a, b) => [box(x, y, 146, 86, kind), T(x + 73, y + 38, a, { bold: true }), T(x + 73, y + 60, b, { size: 12, color: C.muted })].join("\n");
  I.push(cell(182, 66, "ref", "TN, benar", "tamu datang"));
  I.push(cell(332, 66, "key", "FP, biaya 2", "risiko overbooking"));
  I.push(cell(182, 156, "key", "FN, biaya 1", "kamar kosong"));
  I.push(cell(332, 156, "ref", "TP, benar", "batal terdeteksi"));
  I.push(T(330, 274, "Biaya total", { anchor: "end", size: 13 }), M(String.raw`=2\cdot FP+1\cdot FN`, 410, 270, 16));
  write("confusion", frame(W, H, "Confusion matrix dan biaya kesalahan",
    "Baris adalah kelas aktual dan kolom adalah prediksi. TN dan TP adalah prediksi benar. FP, yaitu prediksi batal padahal tamu datang, berbiaya 2 karena berisiko overbooking. FN, yaitu pembatalan yang tidak terdeteksi, berbiaya 1. Biaya total sama dengan 2 kali FP ditambah 1 kali FN.", I));
}
console.log("ok");
