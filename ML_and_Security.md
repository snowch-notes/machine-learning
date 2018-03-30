https://www.safaribooksonline.com/library/view/secure-because-math/9781491954850/video237461.html (2015)

- KDD dataset too old now; historic value only

- model impact (more to less): ground truth (good data) > feature selection > algorithm
- anonymaly detecting difficult (impossible?)
    - host to host network paths - curse of dimensionality
    - no ground truth out of the control of the attacked (normality poisioning)
    - hanlon's razor (aka semantic gap) - never attribute to malice that which is adequately explained by stupidity (lot's of false positives)
- usually not enough ground truths in security data?
- user behavoir anaytics (ueba)
    - e.g. someone failed password too many times or looked at too many files (however, prone to false positives)
    - don't focus on unsupervised
- supervised
    - good sources of (paid) malicious data
    - sources of vetted non-malicious data still a challenge
    - timeliness is a big challenge, models need to be kept up to date
    - areas
       - malware and file have main focus
          - file structure
          - execution behavoir
       - network based
          - who machines are talking to
    
