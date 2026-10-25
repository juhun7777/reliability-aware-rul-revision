# Definitions for revision questions

These separate code definitions, manuscript statements and historical evidence.
Plausible explanations must not be turned into fabricated experimental history.

## Calculation purpose and current checks

The force-resistance fit first models ordinary force-dependent resistance in
the stated initial 0--20000-cycle region. Fitting a straight line to
`x=F**(-1/3), y=R` gives slope alpha and intercept beta by least squares. R²
describes this calibration fit, not RUL prediction accuracy. If R is in mOhm
and F in gf, alpha is in mOhm*gf**(1/3) and beta in mOhm; using N changes alpha.
The original fit input units/gain must be tied to the source calibration record.
The revised training config retains 2.34/0.133 as legacy constants.

Noise diagnostics ask whether a representation preserves its lifetime-related
trend after sensor perturbation. The additional equal-input test perturbs both
R and F, recomputes PHI, and compares the mean drop in absolute RUL correlation:
`drop(raw R) / drop(PHI)`. Standard deviations of 0.5%, 1%, 3% of the initial
100-row sensor means, each repeated ten times over 15 evaluation files, gave
ratios 0.92895, 0.95780, 1.01100. These are comparison results, not noise levels
or a reproduction of the manuscript's 12x result at SNR20dB. Nonpositive or
negligible drops do not define a useful ratio.

SNR20dB means signal power / noise power = 100. For a mean-removed signal,
setting Gaussian noise std to signal std/10 gives that nominal power ratio.
This is not 10% of the sensor mean. The archived raw-only drift/spikes alter
the final ratio and differ from equal-input noise.

Uncertainty/error correlation asks whether high predicted spread accompanies
large actual errors: `Pearson(sigma, abs(prediction-target))`. The nine printed
Table VIII observations give r=0.599200 and descriptive p=0.088158, rather than
0.68 and p<0.05. The latter values are not assigned to a different prediction
run after the fact. Repeated observations from the same specimen require
appropriate grouping when making inferential claims. MC spread by itself does
not establish calibrated confidence or unseen-condition detection.

| Question | Reproducible definition / action |
| --- | --- |
| alpha, beta, R² | The stated model is R=alpha*F^(-1/3)+beta. The manuscript describes initial0--20000cycle regression. The published code uses constants; the fitting dataset, gains and units need their original calibration record. The evidence script can refit declared data without forcing old coefficients. |
| 12x noise robustness | Define the statistic and compare raw R and recomputed PHI from exactly the same perturbed R/F. The prior figure-generation script used additional raw-only drift/spikes, so its comparison is not equal-input evidence. The new script reports correlation drops, including nonpositive/undefined ratios. |
| rho=0.68 | Specify Pearson(sigma_i,abs(prediction_i-target_i)), observation unit and same checkpoint/MC policy. Do not combine a nine-observation table with a separate many-window array. Whole-cycle rounding sensitivity can be bounded. |
| STFT95% | The original source shuffled overlapping windows80:20 and fit its scaler before the split. The revised MLP separates file/parent groups first. Its accuracy belongs to the new split, not the old95% claim. |
| TableVII Statistics | STFT+MLP uses128 spectral+32 temporal statistics. RandomForest statistics have another extractor. The RUL Bi-LSTM uses PHI,R,F,relativeR,cycle, not that32-stat vector. Define or correct the table's input term before re-running its ablation. |
| 0.7A_2 78.6% | Given EOL345100 and remaining73900, observation271200 yields78.5859%. This inversion assumes the filename matches the table row. It is not proof why the row was selected. Simple floor rounding cannot turn an80--90%cut into78.6%. Export exact IDs/rows/cycles in future runs. |
| TableI and refs13--15 | Verify author, title, venue, DOI and the method actually used, not just a similar title. The listed table citations16--19 map to different works in the bibliography. Correct numbering and study-to-method mapping separately. |

Verified primary records useful for bibliography repair (not automatic
replacements for an unsupported SVM/RandomForest claim):

- H.-C. Teng et al., *Study of Contact Degradation in Final Testing for BGA Socket*, EMAP2007, DOI[10.1109/EMAP.2007.4510290](https://doi.org/10.1109/EMAP.2007.4510290). [Co-author university record](https://paper.isu.edu.tw/paper/class_paper_main.php?dep_id=80202&gType=2&lang_type=&type=1&user_id=18102).
- Meng et al., *Investigation on Intermittent Failure Phenomenon of Electrical Connectors Under Repetitive Mechanical Insertion and Withdrawal Operations*, IEEE TCPMT12(9),1533--1541,2022. [Author university record](https://homepage.hit.edu.cn/renwb).
- M. Ahmad and S. K. Sitaraman, *Study of Mechanical Behavior of Compliant Micro-Springs for Next-Generation Probing Applications*, Journal of Electronic Packaging124,411--418,2002. [Author laboratory record](https://caspar.gatech.edu/journal-publications/).
- B. F. Toler, R. A. Coutu Jr., J. W. McBride, *A review of micro-contact physics for microelectromechanical systems (MEMS) metal contact switches*, JMM23,103001,2013. DOI[10.1088/0960-1317/23/10/103001](https://doi.org/10.1088/0960-1317/23/10/103001). [Author institutional record](https://scholar.afit.edu/facpub/4093/).
- H. Qiu, H. Wang, F. Ke, *Instability of Contact Resistance in MEMS and NEMS DC Switches under Low Force: the Role of Alien Films on the Contact Surface*, Sensors13,16360--16371,2013. DOI[10.3390/s131216360](https://www.mdpi.com/1424-8220/13/12/16360).

The last two are different real papers. Do not combine Toler's authors with
Qiu's title or an unrelated IEEE journal/volume/pages. Exact unmatched records
are marked for correction; search failure alone does not prove nonexistence.

## Proposed distinct replacements for references 13 to 15

The proposed replacements use separate real studies rather than merging with
existing references. They do not preserve unsupported Random Forest/FEA labels;
the surrounding prose and Table I method columns must follow each study.

- **[13]** X. Cheng, K. Lv, G. Liu, J. Qiu, *Intermittent Fault Modeling and RUL
  Prediction for Degraded Electrical Connectors in Vibration Environments*,
  IEEE TCPMT 12(5), 769--777, 2022.
  DOI [10.1109/TCPMT.2022.3167638](https://doi.org/10.1109/TCPMT.2022.3167638).
  The paper abstract describes a health indicator and GA-optimized ELM for RUL.
- **[14]** X. Cheng et al., *RUL Prediction Method for Electrical Connectors With
  Intermittent Faults Based on an Attention-LSTM Model*, IEEE TCPMT 13(5),
  628--637, 2023.
  DOI [10.1109/TCPMT.2023.3282616](https://doi.org/10.1109/TCPMT.2023.3282616).
  The paper abstract describes contact-resistance/intermittent-fault features
  and attention-LSTM RUL prediction.
- **[15]** H. Qiu, H. Wang, F. Ke, *Instability of Contact Resistance in MEMS and
  NEMS DC Switches under Low Force: the Role of Alien Films on the Contact
  Surface*, Sensors 13(12), 16360--16371, 2013.
  DOI [10.3390/s131216360](https://doi.org/10.3390/s131216360).
  This supports the contact-physics discussion, not an RUL benchmark row.

Bibliographic metadata was checked against official DOI registration records;
the Cheng method summaries use the publicly available research-paper abstracts.
These are reference correction proposals, not edits to the submitted manuscript.
