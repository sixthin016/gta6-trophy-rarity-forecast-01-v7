# gta6-trophy-rarity-forecast-01 — v7 build notes

- Kompetisi: forecast 55 trophy GTA VI (Exophase) di 2026-11-26 00:00 UTC,
  5 kuantil/trophy, skor pinball loss skala logit. Hadiah $250, top 20
  (1st $50, 2nd $35, 3rd $25, 4-10 $10, 11-20 $7). Deadline entri
  21 Nov 2026 11:30 UTC = 19:30 WITA. Kuota dicek 21 Nov ~15:30 WITA:
  can_submit=true, day cap 6, taken 2, left 4, hard limit left 35/40,
  entries_made 5 (v1, v2, v3r, v5, v6).

## Diagnosis: kenapa v6 rank 1035 padahal proxy-nya 0,0794

- Loss live v6 (diukur langsung dari leaderboard 21 Nov ~15:30 WITA):
  **0,148716**, rank 1035/8982, typical miss 0,05 pp, coverage 0,9455.
  Proxy v6 bilang 0,0794 (perkiraan rank ~110-130). Proxy-nya salah
  kalibrasi, bukan modelnya yang tiba-tiba rusak.
- Kalibrasi ulang (tools/calib_v7.py): cari family kebenaran yang
  mereproduksi keempat anchor board nyata secara bersamaan --
  persistence 1,51, v5 1,85, v2 0,139*, v6 0,148716. Family: newcomer
  rate per trophy = c * r^beta * exp(sigma*z), di-rescale ke total
  4.276 earners. Hasil: beta hampir tidak berpengaruh; sigma yang
  menentukan. Sigma 1,25 mereproduksi anchor dengan fit log-error
  0,0032 (pers 1,4485, v5 1,7917, v6 0,1456). Proxy lama memakai
  dispersi efektif sigma ~0,3-0,5 -- JAUH terlalu sempit. Itulah
  kenapa v6 (fan logit sigma ~0,23) terlihat bagus di proxy tapi
  kalah di board nyata: fan-nya terlalu sempit untuk dispersi nyata.
- *Koreksi dua angka di notes lama: v1/v2 0,168/0,139 TIDAK MUNGKIN
  benar -- v1 dan v2 byte-identik di practice (diverifikasi cmp) dan
  fan-nya normal logit sigma=1,0 yang lantai pinball-nya saja ~0,197.
  Kedua angka itu salah baca board. v2 tidak dipakai sebagai anchor
  fit (hanya referensi); kalibrasi memakai pers/v5/v6 saja.
- Papan atas practice (rank 1-3 loss 0,000001, cutoff top 20 0,0203)
  TIDAK bisa dicapai model jujur: klaster jujur 0,08-0,15. Entri
  puncak memprediksi jawaban practice nyaris persis -- itu hanya
  membantu papan practice, BUKAN grade final (26 Nov) yang membayar
  dan memakai histori bergerak. v7 memperbaiki model jujur; tidak
  mengejar para peekers.

## Desain v7 (perubahan terhadap v6, semua terukur)

1. **NEW_AGE_H 0,20 -> 0,15** (tools/exp_fan.py, tools/exp_newage.py):
   median pendatang baru sedikit lebih rendah. Menang KONSISTEN di
   semua sigma recalibrated (0,5/1,0/1,25/1,5): s=1,0 0,1294->0,1230,
   s=1,25 0,1478->0,1384, s=1,5 0,1640->0,1512; proxy lama hanya
   +0,0009 (0,0802->0,0811). Grid halus: 0,08/0,10/0,12/0,15/0,18/
   0,20 -> optimum basin dangkal 0,12-0,15, mean terbaik di 0,15.
2. **Fan median-preserving**: v6 menggambar lam sebagai
   lam_bar*exp(S_ALPHA*z - 0,5*S_ALPHA^2) -- menjaga MEAN tapi
   menggeser MEDIAN fan ke bawah exp(-0,5*S_ALPHA^2). Untuk forecast
   kuantil, q50 harus duduk tepat di median dilusi; v7 membuang
   koreksinya (lam = lam_bar*exp(S_ALPHA*z)).
3. **DITOLAK oleh pengukuran (tidak ada di v7)**:
   - Melebarkan fan (S_MULT 0,15->0,35, S_ALPHA 0,30->0,80): kalah
     di SEMUA suite, proxy lama maupun recalibrated.
   - Melebarkan hanya kuantil atas (x1,5 / x2,0 di ruang logit):
     kalah di mana-mana.
   - Memasukkan growth kohort terlacak ke median flat (gamma 0,15/
     0,5/1,0; tools/calib_growth.py): fit anchor MEMBURUK vs
     gamma=0 -- pemain terlacak praktis tidak menambah earners yang
     terlihat di dalam horizon. Perilaku v6 (growth dihitung tapi
     dibuang pemanggil flat path) ternyata BENAR dan dipertahankan.
   - GROWTH tidak teridentifikasi di jalur flat (flat_median
     menghitungnya tapi pemanggil tidak memakai e_t) -- dibiarkan
     0,15 untuk jalur moving.
4. **Jalur moving TIDAK DIUBAH** (desain v6 dipertahankan penuh):
   yang membayar adalah grade final, dan di sana v6 setara model
   jujur terbaik. Median flat v7 tetap konsisten-agregat: total
   earners baru prediksi (median) 3.313/3.297 vs kebenaran 3.333.

## Hasil validasi (tools/backtest_v7.py, N_MC=3000)

| suite | v7 | v6 | v4 | v5 | persistence |
|---|---|---|---|---|---|
| (a) practice-proxy lama (13) | **0,0809** | 0,0794 | 0,1135 | 1,7459 | 1,4034 |
| (b) recalibrated sigma 0,5/1,0/1,25 (18) | **0,1175** | 0,1213 | 0,1458 | 1,7738 | 1,4310 |
| &nbsp;&nbsp;sigma 0,5 | 0,0911 | 0,0906 | 0,1219 | 1,7533 | 1,4107 |
| &nbsp;&nbsp;sigma 1,0 | 0,1230 | 0,1276 | 0,1499 | 1,7764 | 1,4337 |
| &nbsp;&nbsp;sigma 1,25 | 0,1383 | 0,1456 | 0,1657 | 1,7917 | 1,4485 |
| (c) v5-style final-like (8) | 0,7972 | 0,7960 | 1,7800 | 0,5797 | 1,5518 |
| (d) v4-style synthetic (praktik+final+CV) | 0,1785 | 0,1778 | 0,1955 | 1,1197 | 1,0652 |

Perkiraan loss live practice v7 ~0,138 (sigma 1,25) vs v6 0,1487 --
rank perkiraan ~800 dari 8.982, masih jauh dari top 20 (0,0203) yang
tidak terjangkau model jujur. Suite (c)/(d): v7 vs v6 beda <=0,0016
(noise; jalur moving tidak berubah). Regresi di proxy lama hanya
+0,0015 -- harga dari median yang lebih rendah, dibayar kembali
3-5% di dunia yang kalibrasinya cocok dengan board nyata.

## Batasan / risiko

- Kalibrasi mengasumsikan dispersi lognormal beta=1; bila dispersi
  nyata lebih besar (sigma >1,5), skor live v7 bisa lebih buruk dari
  0,14; bila lebih kecil, bisa lebih baik. Median (NEW_AGE 0,15)
  kokoh di semua sigma yang diuji.
- main.py ditulis fresh di sesi ini (bukan salin v6) demi authorship
  transcript: tools/calib_v7.py, tune_v7.py, exp_fan.py,
  calib_growth.py, exp_newage.py, backtest_v7.py, NOTES-v7.md dan
  README.md semuanya ditulis fresh; main.py distruktur ulang dengan
  perubahan v7. Jalur moving dipertahankan perilakunya sama dengan v6.
- Deterministik (2 run identik). Runtime ~2,6-3,4 dtk/kasus (< 8 dtk).
  Output 2.772 byte (< 65.536).
