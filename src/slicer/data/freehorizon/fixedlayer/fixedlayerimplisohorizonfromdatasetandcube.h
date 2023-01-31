

#ifndef __FIXEDLAYERIMPLISOHORIZONFROMDATASETANDCUBE__
#define __FIXEDLAYERIMPLISOHORIZONFROMDATASETANDCUBE__

#include <QString>

#include <fixedlayersfromdatasetandcube.h>

class FixedLayerImplIsoHorizonFromDatasetAndCube : public FixedLayersFromDatasetAndCube//, public IData, public StackableData, public IsoChronProvider
{
public:
	FixedLayerImplIsoHorizonFromDatasetAndCube(QString dirPath, QString dirName, QString seismicName, WorkingSetManager *workingSet,
						const FixedLayersFromDatasetAndCube::Grid3DParameter &params, QObject *parent = 0);

	virtual ~FixedLayerImplIsoHorizonFromDatasetAndCube();
	void getImageForIndex(long newIndex,
			CUDAImagePaletteHolder* attrCudaBuffer, CUDAImagePaletteHolder* isoCudaBuffer);
	// use rgb1
	bool getImageForIndex(long newIndex,
			QByteArray& attrBuffer, QByteArray& isoBuffer);

	void setCurrentImageIndexInternal(long newIndex);
	void setCurrentImageIndex(long newIndex);
	QString getIsoFileFromIndex(int index);
	QColor getHorizonColor();
	QString dirPath() { return m_dirPath; }
	void setHorizonColor(QColor col);
	QString getObjFile(int index) const	override { return ""; }
	bool enableSlicePropertyPanel() override { return true; }
	QString propertyPanelType() override { return "default"; }





private:
	void loadIsoAndAttribute(QString attribute);
	QString getAttributFileFromIndex(int index);

	QString m_dirPath = "";
	QString m_dirName = "";
	QString m_seismicName = "";
	std::vector<QString> m_attributNames;
	std::vector<QString> m_isoNames;


};







#endif
