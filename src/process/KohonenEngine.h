#ifndef SRC_PROCESS_KOHONENENGINE_H_
#define SRC_PROCESS_KOHONENENGINE_H_

#include <vector>

// 1D TMAP
class KohonenEngine {
public:
	KohonenEngine(long DimInput, long ne_tailleCarte);
	virtual ~KohonenEngine();

	bool trainTmap(float *exemples, long NbExempl, long ne_nbCycles, long ne_tailleVoisin,
			float sigmamax, float sigmamin, float epsilmax,float epsilmin, long voisinag,
			long typenorm);

	long applyTmap(float* example, bool* ok) const;

private:
	static void NeNormer(float *exemples, float *exempmax, float *exempmin,
	            long diminput, long nbexempl, long typenorm);

	std::vector<float> m_tmap;

	long m_exampleSize, m_tmapSize;

};

#endif
