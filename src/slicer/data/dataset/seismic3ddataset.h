#ifndef Seismic3DDataset_H
#define Seismic3DDataset_H

#include <QMutex>
// #include "fileio2.h"
#include "seismic3dabstractdataset.h"

class Seismic3DDataset: public Seismic3DAbstractDataset {
Q_OBJECT
public:
	Seismic3DDataset(SeismicSurvey *survey,const QString &name, WorkingSetManager *workingSet,
			CUBE_TYPE type = CUBE_TYPE::Seismic, QString idPath="", QObject *parent = 0);
	virtual ~Seismic3DDataset();

	//IData
	virtual IGraphicRepFactory* graphicRepFactory();

	template<typename DataType>
	void readInlineBlock(DataType *output, int z0, int z1, bool returnBigEndian=true) const;
	template<typename DataType>
	void readTraceBlock(DataType *output, int y0, int y1, int z, bool returnBigEndian=true) const;
	template<typename DataType>
	void readSubTrace(DataType *output, int x0, int x1, int y, int z, bool returnBigEndian=true) const;
	template<typename DataType>
	void readTraceBlockAndSwap(DataType *output, int y0, int y1, int z) const;
	template<typename DataType>
	void readSubTraceAndSwap(DataType *output, int x0, int x1, int y, int z) const;

	// dimVHint is there because inri::Xt does not read dimV correctly
	// dimVHint < 1 mean no hint, dimVHint > 0 mean : use dimV = dimVHint if header dimV==1 and dimSample % dimVHint == 0
	// this can be risky because the rest of the program expect xt files to give correct dimSample and dimV
	void loadFromXt(const std::string &path, int dimVHint=-1) override;
	void loadInlineXLine(CUDAImagePaletteHolder *cudaImage, SliceDirection dir,
			unsigned int z, unsigned int c=0, SpectralImageCache* cache=nullptr) override;
	void loadRandomLine(CUDAImagePaletteHolder *cudaImage,
			const QPolygon& randomLine, unsigned int c=0, SpectralImageCache* cache=nullptr) override;

	std::string path() const {
		return m_path;
	}

	size_t headerLength() const {
		return m_headerLength;
	}

	virtual QVector2D minMax(int channel, bool forced=false) override;

	template<typename InputType, typename OutputType>
	static void swapAndCopyTab(InputType* tabIn, OutputType* tabOut, std::size_t size);
	template<typename InputType>
	static void swapTab(InputType* tabIn, std::size_t size);

	virtual bool writeRangeToFile(const QVector2D& range) override;
private:
	void releaseContent();
private:
	typedef struct ChannelCouple {
		CUDAImagePaletteHolder* image;
		unsigned int c;
	} ChannelCouple;

	template<typename DataType>
	struct ReadInlineBlockKernel {
		static void run(const Seismic3DDataset* obj, QList<ChannelCouple> image, int z, void* cache);
	};
	template<typename DataType>
	struct ReadXLineBlockKernel {
		static void run(const Seismic3DDataset* obj, QList<ChannelCouple> image, int y, void* cache);
	};
	template<typename DataType>
	struct ReadRandomLineKernel {
		static void run(const Seismic3DDataset* obj, QList<ChannelCouple> image, const QPolygon& poly, void* cache);
	};

	Seismic3DDatasetGraphicRepFactory *m_repFactory;
	mutable QMutex m_lock;
	FILE *m_currentFile;
	size_t m_headerLength;
	std::string m_path;
	// FILEIO2 *pfileio2;

};
Q_DECLARE_METATYPE(Seismic3DDataset*)

#include "seismic3ddataset.hpp"
#endif
