# Published CIE cooling tables used by `cgols_paper2_cooling_checks.py`

Reference data for the ASGARD Paper II checks of the model-B cooling curve. They were read as plain-text
tables from the authors' pages on 25 November 2026 and converted to JSON without changing any value.
Each file keeps the authors' own normalisation, and the check script converts every curve to per n_H².

| File | Source | Normalisation as published |
|---|---|---|
| `gf12_element_cie.json` | Gnat & Ferland 2012, ApJS 199, 20, doi:10.1088/0067-0049/199/1/20. Tables 1A–1AD, last column (element CIE cooling efficiency), from http://wise-obs.tau.ac.il/~orlyg/ion_by_ion/datafile1{A..AD}.txt | Per n_e n_element (erg cm³ s⁻¹), Cloudy 08.00 CIE ion fractions |
| `gs07_cie_cooling.json` | Gnat & Sternberg 2007, ApJS 168, 213, doi:10.1086/509786. CIE cooling efficiencies for Z = 10⁻³, 10⁻², 10⁻¹, 1, 2 Z☉, from http://wise-obs.tau.ac.il/~orlyg/cooling/CIEcool/tab13.txt | Per n_e n_H (their Sect. 2.2: dQ = −n_e n_H Λ) |
| `sd93_solar_cie.json` | Sutherland & Dopita 1993, ApJS 88, 253, doi:10.1086/191823. Solar CIE table `m-00.cie`, from https://www.mso.anu.edu.au/~ralph/data/cool/ | log λ_net at n_H = 1 cm⁻³, i.e. per n_H². log λ_norm is per n_e n_t (ABOUT.txt: "lambda norm = lambda net / (ne nt)") |
| `schure09_table2.json` | Schure et al. 2009, A&A 508, 751, doi:10.1051/0004-6361/200912495. Table 2 (arXiv:0909.5204) | Λ_N per n_e n_H; Λ_hd = (n_e/n_H) Λ_N per n_H² (their Eq. 1). This is the original SPEX curve; Štofanová et al. 2021 (doi:10.1051/0004-6361/202037768) revised its 10⁵ K peak downwards |

The abundance sets differ:

| Table | Abundances |
|---|---|
| GF12 element tables | Per element; combined here with Asplund et al. (2009) number abundances |
| GS07 | Asplund et al. (2005), with Ne from Drake & Testa (2005) |
| SD93, Schure09 | Anders & Grevesse (1989) |
