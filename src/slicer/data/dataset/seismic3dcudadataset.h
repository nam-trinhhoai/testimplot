#ifndef Seismic3DCUDADataset_H
#define Seismic3DCUDADataset_H

/**
 * Modification code to adapt to dataset sampleType not done
 * Modification code to adapt to dataset channel not done
 *
 * It was not done because, I do not know this code, code that must contain a lot of cuda code.
 * I think it will take too long and the command was to avoid too long modifications
 *
 * Armand Sibille L0483271 19/02/2021
 */

#include "seismic3dabstractdataset.h"

class Seismic3DCUDADataset: public Seismic3DAbstractDataset {
Q_OBJECT
public:
	Seismic3DCUDADataset(SeismicSurvey *survey, const QString &name,WorkingSetManager *workingSet,
			CUBE_TYPE type = CUBE_TYPE::Seismic, QString idPath="", QObject *parent = 0);
	virtual ~Seismic3DCUDADataset();

	//IData
	virtual IGraphicRepFactory* graphicRepFactory();

	// dimVHint is there because inri::Xt does not read dimV correctly
	// dimVHint < 1 mean no hint, dimVHint > 0 mean : use dimV = dimVHint if header dimV==1 and dimSample % dimVHint == 0
	// this can be risky because the rest of the program expect xt files to give correct dimSample and dimV
	void loadFromXt(const std::string &path, int dimVHint=-1) override;
	void loadInlineXLine(CUDAImagePaletteHolder *cudaImage, SliceDirection dir,
			unsigned int z, unsigned int c=0, SpectralImageCache* cache=nullptr) override;
	void loadRandomLine(CUDAImagePaletteHolder *cudaImage,
			const QPolygon& randomLine, unsigned int c=0, SpectralImageCache* cache=nullptr) override;

	short* cudaBuffer();
	virtual QVector2D minMax(int channel, bool forced=false) override;

	virtual bool writeRangeToFile(const QVector2D& range) override;

private:
	void releaseContent();
private:
	template<typename InputType>
	struct InitContentKernel {
		static std::size_t run(Seismic3DCUDADataset* obj, long trueDimV, FILE* fp);
	};

	bool m_cudaLoaded;

	std::string m_xtFile;
	short *m_content;
	short *m_cudaBuffer;
	Seismic3DDatasetGraphicRepFactory *m_repFactory;
};
Q_DECLARE_METATYPE(Seismic3DCUDADataset*)

#endif
