#ifndef __FREEHORIZONQMANAGER__
#define __FREEHORIZONQMANAGER__

#include "grid2d.h"
#include "viewutils.h"

#include <vector>
#include <QString>
#include <QColor>
#include <QIcon>

class FreeHorizonQManager {
public:
	static std::vector<QString> getListName(QString path);
	static std::vector<QString> getListPath(QString path);
	static std::vector<QString> getDataSet(QString horizonPath);
	static QString getPrefixFromFile(QString filename);
	static QString getRgtDataSetNameFromPath(QString path);
	static QString getDataSetNameFromPath(QString path);
	static QColor loadColorFromPath(const QString &path, bool *ok);
	static bool saveColorToPath(const QString &path, const QColor &color);

	template<typename InputIterator>
	static bool getCroppedBufferFromGrids(const Grid2D &inputGrid,
			const Grid2D &outputGrid, InputIterator inputBufBegin,
			InputIterator inputBufEnd, std::vector<float> &outputBuffer);
	template<typename InputIterator, typename OutputIterator>
	static bool getCroppedBufferFromGrids(const Grid2D &inputGrid,
			const Grid2D &outputGrid, InputIterator inputBufBegin,
			InputIterator inputBufEnd, OutputIterator outputBufBegin,
			OutputIterator outputBufEnd);

	static std::vector<QString> getAttributData(QString& path);
	static std::vector<QString> getAttributPath(QString& path);


	static QIcon getHorizonIcon(QColor color, SampleUnit sampleUnit, int size = 32);
	static QIcon getHorizonIcon(QString path, int size = 32);

	static QIcon getDataSetIcon(QString path);

	static int getIsoFromDirectory(QString dir);
	static QString getSizeOnDisk(QString path);
};

#include "freeHorizonQManager.hpp"

#endif
