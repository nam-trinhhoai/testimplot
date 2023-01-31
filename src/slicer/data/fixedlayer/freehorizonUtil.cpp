
#include <malloc.h>
#include <QDebug>
#include <utility>
#include <CubeIO.h>
#include <freehorizonUtil.h>

std::string freeHorizonSaveWithoutTdebAndPasech(std::string filename, float *data, int dimy, int dimz, float tdeb, float pasech)
{
	float *data0 = (float*)calloc((long)dimy*dimz, sizeof(float));
	if ( data0 == nullptr ) { qDebug() << "error in freeHorizonSaveWithoutTdebAndPasech"; return "error"; }

	for (long add=0; add<dimy*dimz; add++)
		data0[add] = ( data[add] - tdeb ) / pasech;

	// const murat::io::InputOutputCube<float> *cube = murat::io::createCube<float>(filename, dimy, dimz, 1, murat::io::SampleType::E::FLOAT32, true);
	// delete cube;
	// cube = murat::io::openCube<float>(filename);
	const murat::io::InputOutputCube<float> *cube = murat::io::openOrCreateCube<float>(filename, dimy, dimz, 1, 0, murat::io::SampleType::E::FLOAT32, true);
	cube->writeSubVolume(0, 0, 0, data0, dimy, dimz, 1);
	free(data0);
	delete cube;
	return "ok";
}

std::string freeHorizonRead(std::string filename, void *data)
{
	const murat::io::InputOutputCube<float> *cube = murat::io::openCube<float>(filename);
	murat::io::CubeDimension dims = cube->getDim();
	int dimy = dims.getI();
	int dimz = dims.getJ();
	cube->readSubVolume(0, 0, 0, dimy, dimz, 1, (float*)data);
	delete cube;
	return "ok";
}

std::pair<int, int> freeHorizonGetDims(std::string filename)
{
	const murat::io::InputOutputCube<float> *cube = murat::io::openCube<float>(filename);
	murat::io::CubeDimension dims = cube->getDim();
	int dimy = dims.getI();
	int dimz = dims.getJ();
	delete cube;
	return std::make_pair(dimy, dimz);
}


















/*
QString freeHorizonWriteDims(QString path, int dimy, int dimz)
{
	QString filename = path + FREEHORIZON_JSONFILENAME;
	QFile file(filename);
	if (!file.open(QIODevice::WriteOnly)) {
		qDebug() << "freeHorizonWriteDims : cannot save dims, file not writable";
		return "error";
	}
	QJsonObject obj;
	QJsonArray qjDims;

	qjDims.append(dimy);
	qjDims.append(dimz);

	obj.insert(freehorizondimsKey, qjDims);
	QJsonDocument doc(obj);
	file.write(doc.toJson());

	return "success";
}

std::pair<int, int> freeHorizonGetDims(QString path)
{
	int dimy = 0;
	int dimz = 0;

	QString filename = path + FREEHORIZON_JSONFILENAME;
	QFile file(filename);
	if (!file.open(QIODevice::ReadOnly)) {
		qDebug() << "freeHorizonWriteDims : cannot read dims, file not readable";
		return std::make_pair(dimy, dimz);
	}
	QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
	if (!doc.isObject()) {
		qDebug() << "freeHorizonWriteDims : cannot read dims, root is not a json object";
		return std::make_pair(dimy, dimz);
	}

	QJsonObject rootObj = doc.object();
	if (rootObj.contains(freehorizondimsKey) && rootObj.value(freehorizondimsKey).isArray()) {
		QJsonArray array = rootObj.value(freehorizondimsKey).toArray();
		dimy = array[0].toInt(0);
		dimz = array[1].toInt(0);
	}

	return std::make_pair(dimy, dimz);
}
*/
