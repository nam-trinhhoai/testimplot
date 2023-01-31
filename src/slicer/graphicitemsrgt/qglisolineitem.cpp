#include "qglisolineitem.h"

#include <QTransform>
#include <QOpenGLBuffer>
#include <QMatrix4x4>
#include <QOpenGLShaderProgram>
#include <iostream>
#include <QGraphicsSceneMouseEvent>
#include <cuda_runtime_api.h>
#include "cuda_volume.h"
#include "cudaimagepaletteholder.h"
#include "sampletypebinder.h"

QGLIsolineItem::QGLIsolineItem(const IGeorefImage * const transfoProvider,
		CUDAImagePaletteHolder *isoSurface, int defaultExtractionWndindow,
		SliceDirection dir, QGraphicsItem *rgtProvider) :
		QAbstractGLGraphicsItem(rgtProvider) {

	m_dir = dir;
	m_isoSurface = isoSurface;
	m_initialized = false;

	if (dir == SliceDirection::Inline) {
		m_internalBufferSize = isoSurface->width() * 2 * sizeof(float);
		m_backingBuffer = new float[isoSurface->width() * 2];
	} else {
		m_internalBufferSize = isoSurface->height() * 2 * sizeof(float);
		m_backingBuffer = new float[isoSurface->height() * 2];
	}

	m_program = new QOpenGLShaderProgram(this);
	m_worldExtent = transfoProvider->worldExtent();
	m_matrix=transfoProvider->imageToWorldTransformation();
	m_integrationWindow = defaultExtractionWndindow;
	QColor c(Qt::yellow);
	m_color = QVector4D(c.redF(), c.greenF(), c.blueF(), 0.5f);
	m_currentPos = 0;

	cudaMalloc(&m_cudaBuffer, m_internalBufferSize);
	m_needInternalBufferUpdate = true;
}

QGLIsolineItem::~QGLIsolineItem() {
	cudaFree(m_cudaBuffer);
	delete[] m_backingBuffer;
}
void QGLIsolineItem::updateSlice(int value) {
	m_currentPos = value;
	m_needInternalBufferUpdate = true;
	update();
}

void QGLIsolineItem::updateRGTPosition() {
	m_needInternalBufferUpdate = true;
	update();
}

template<typename InputType>
struct IsoLineExtractKernel {
	static void run(void* cudaPtr, int w, int h, unsigned int pos, unsigned int dir, float* cudaBuffer) {
		isoLineExtract((InputType*) cudaPtr, w, h,
							pos, dir, cudaBuffer);
	}
};

void QGLIsolineItem::updateInternalBuffers() {
	if (m_initialized) {
		//must be kept outside the following as it's loking the buffer...
		int w = m_isoSurface->width();
		int h = m_isoSurface->height();
		SampleTypeBinder binder(m_isoSurface->sampleType());
		m_isoSurface->lockPointer();
		if (m_dir == SliceDirection::Inline)
			binder.bind<IsoLineExtractKernel>(m_isoSurface->cudaPointer(), w, h,
					m_currentPos, 0, (float*) m_cudaBuffer);
		else
			binder.bind<IsoLineExtractKernel>(m_isoSurface->cudaPointer(), w, h,
					m_currentPos, 1, (float*) m_cudaBuffer);
		m_isoSurface->unlockPointer();
		cudaMemcpy(m_backingBuffer, m_cudaBuffer, m_internalBufferSize,
				cudaMemcpyDeviceToHost);
		m_vertexBuffer.write(0, (const char*) m_backingBuffer,
				m_internalBufferSize);
		m_needInternalBufferUpdate = false;
	}
	update();
}

void QGLIsolineItem::setColor(QColor c) {
	m_color = QVector4D(c.redF(), c.greenF(), c.blueF(), 0.5f);
	update();
}

void QGLIsolineItem::updateWindowSize(unsigned int w) {
	m_integrationWindow = w;
	m_needInternalBufferUpdate = true;
	update();
}

void QGLIsolineItem::initializeGL() {
	initShaders();

	m_vertexBuffer.create();
	m_vertexBuffer.bind();
	m_vertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	m_vertexBuffer.allocate(m_internalBufferSize);
	m_vertexBuffer.release();

	m_initialized = true;
}

void QGLIsolineItem::initShaders() {
	if (!loadProgram(m_program, ":shaders/common/common.vert",
			":shaders/common/simpleColor.frag"))
		qDebug() << "Failed to initialize shaders";
}

void QGLIsolineItem::drawGL(const QMatrix4x4 &viewProjectionMatrix,
		const QRectF &exposed, int width, int height, int dpiX, int dpiY) {
	if (!m_initialized)
		initializeGL();

	glClearColor(0, 0, 0, 0);

	m_vertexBuffer.bind();
	if (m_needInternalBufferUpdate) {
		updateInternalBuffers();
		m_vertexBuffer.release();
		return;
	}

	int length = m_isoSurface->width();
	if (m_dir == SliceDirection::XLine)
		length = m_isoSurface->height();

	if (!m_program->isLinked()) {
		if (!m_program->link())
			return;
	}

	// program setup
	m_program->bind();

	m_program->setUniformValue("viewProjectionMatrix", viewProjectionMatrix);
	m_program->setUniformValue("color", m_color);

	QMatrix4x4 matrix=m_matrix;
	m_program->setUniformValue("transfoMatrix", matrix);

	m_program->enableAttributeArray("vertexPosition");
	m_program->setAttributeBuffer("vertexPosition", GL_FLOAT, 0, 2);

	glLineWidth(2.0f);
	glDrawArrays(GL_LINE_STRIP, 0, length - 1);

	glLineWidth(1.0f);
	glEnable(GL_LINE_STIPPLE);
	glLineStipple(1, 0x00FF);

	//Top Line
	matrix.translate(0, m_integrationWindow / 2.0f);
	m_program->setUniformValue("transfoMatrix", matrix);
	glDrawArrays(GL_LINE_STRIP, 0, length - 1);

	//Bottom Line
	matrix=m_matrix;
	matrix.translate(0, -(m_integrationWindow / 2.0f));
	m_program->setUniformValue("transfoMatrix", matrix);
	glDrawArrays(GL_LINE_STRIP, 0, length - 1);
	glDisable(GL_LINE_STIPPLE);

	m_vertexBuffer.release();
	m_program->release();
}

