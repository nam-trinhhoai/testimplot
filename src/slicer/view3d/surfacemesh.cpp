#include "surfacemesh.h"
#include <Qt3DCore/QBuffer>
#include <Qt3DCore/QAttribute>
#include <Qt3DCore/QGeometry>
#include <QFileInfo>

#include <cstring>

#include <math.h>
#include <cmath>
#include <chrono>
#include "qt3dhelpers.h"

#include "surfacegenerator.h"
#include "sampletypebinder.h"


SurfaceMesh::SurfaceMesh( Qt3DCore::QNode *parent)
    : Qt3DRender::QGeometryRenderer(parent)
    , m_geometry(new Qt3DCore::QGeometry())
    , m_positionAttribute(new Qt3DCore::QAttribute())
    , m_vertexBuffer(new Qt3DCore::QBuffer())
	, m_textureAttribute(new Qt3DCore::QAttribute())
    , m_texBuffer(new Qt3DCore::QBuffer())
	, m_indexAttribute(new Qt3DCore::QAttribute())
    , m_indexBuffer(new Qt3DCore::QBuffer())
	, m_normalAttribute(new Qt3DCore::QAttribute())
	, m_normalBuffer(new Qt3DCore::QBuffer())
{

	//m_vertexBuffer->setType(Qt3DCore::QBuffer::VertexBuffer);
	m_positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
	m_positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
//	m_positionAttribute->setDataType(Qt3DCore::QAttribute::Float);
	m_positionAttribute->setBuffer(m_vertexBuffer);

	m_positionAttribute->setVertexSize(3);
	m_positionAttribute->setByteOffset(0);
	m_positionAttribute->setByteStride(3 * sizeof(float));
	m_positionAttribute->setCount(0);


//	m_texBuffer->setType(Qt3DCore::QBuffer::VertexBuffer);
	m_textureAttribute->setName(Qt3DCore::QAttribute::defaultTextureCoordinateAttributeName());
	m_textureAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
//	m_textureAttribute->setDataType(Qt3DCore::QAttribute::Float);
	//m_textureAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
	m_textureAttribute->setBuffer(m_texBuffer);
	m_textureAttribute->setVertexSize(2);
	m_textureAttribute->setByteOffset(0);
	m_textureAttribute->setByteStride(0);
	m_textureAttribute->setCount(0);

//	m_normalBuffer->setType(Qt3DCore::QBuffer::VertexBuffer);
	m_normalAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
	m_normalAttribute->setBuffer(m_normalBuffer);
//	m_normalAttribute->setDataType(Qt3DCore::QAttribute::Float);
	m_normalAttribute->setVertexSize(3);
	m_normalAttribute->setByteOffset(0);
	m_normalAttribute->setByteStride(0);
	m_normalAttribute->setCount(0);
	m_normalAttribute->setName(Qt3DCore::QAttribute::defaultNormalAttributeName());

//    m_indexBuffer->setType(Qt3DCore::QBuffer::UniformBuffer);
    m_indexAttribute->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
    m_indexAttribute->setBuffer(m_indexBuffer);
    m_indexAttribute->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
    //m_indexAttribute->setVertexSize(1);
    m_indexAttribute->setByteOffset(0);
    m_indexAttribute->setByteStride(3 * sizeof(unsigned int));
    m_indexAttribute->setCount(0);

    m_geometry->addAttribute(m_positionAttribute);
    m_geometry->addAttribute(m_normalAttribute);
    m_geometry->addAttribute(m_textureAttribute);
    m_geometry->addAttribute(m_normalAttribute);

    m_geometry->addAttribute(m_indexAttribute);

    setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);//Triangles);//Lines
    setGeometry(m_geometry);

    QObject::connect(this, &SurfaceMesh::dimensionsChanged,
                     this, &SurfaceMesh::updateGeometry);

    QObject::connect(this, &SurfaceMesh::transformChanged,
                     this, &SurfaceMesh::updateGeometry);

    m_steps = 10.0f;

    m_nullValue = -9999.0;
    m_nullValueActive = false;
    m_internalNullValue = -9999.0;
}



SurfaceMesh::~SurfaceMesh()
{
}

void SurfaceMesh::setDimensions(const QVector2D dimensions)
{
    if (m_dimensions == dimensions && m_steps==m_processedSteps)
        return;
    m_dimensions = dimensions;
    emit dimensionsChanged();
}

QVector2D SurfaceMesh::dimensions() const
{
    return m_dimensions;
}


void SurfaceMesh::setTransform(const QMatrix4x4& transform) {
    if (m_transform == transform)
        return;
    m_transform = transform;
    emit transformChanged();
}

QMatrix4x4 SurfaceMesh::transform() const {
	return m_transform;
}


void SurfaceMesh::updateGeometry()
{
	m_processedSteps = m_steps;
	long s = m_steps;

	//s = (long)ceil(sqrt( (double)m_dimensions.x()*(double)m_dimensions.y()*2.0*3.0*4.0 /2.0e9));
	//fprintf(stderr, "++++ scale factor: %d\n", s);

	// change matrix to keep same shape
	QMatrix4x4 scaleMatrix;
	scaleMatrix.scale(s, 1, s);
	QMatrix4x4 transform = m_transform * scaleMatrix;

	int dimx = (int)m_dimensions.x();
	int dimz = (int)m_dimensions.y();

	/*if( dimx % m_steps != 0  || dimz % m_steps != 0)
	{
		qDebug() << " la division n'est pas entiere";
	}*/

	m_width = (m_dimensions.x()-1)/s+1;
	m_height = (m_dimensions.y()-1)/s+1;

}


void SurfaceMesh::computeNormals( )//float* rawVertexArray)//float *vertices, float * normals,unsigned int *indices)
{
	const auto *indices = reinterpret_cast< const uint*>(m_indexBuffer->data().constData());
//	auto *normals = reinterpret_cast<float*>(m_normalBuffer->data().data());
	const auto *rawVertexArray = reinterpret_cast<const float*>(m_vertexBuffer->data().constData());

	 int nbindice = m_indexAttribute->count();
	 int nbpoint = m_positionAttribute->count();


	 QByteArray normalBufferData;
	 normalBufferData.resize(3*nbpoint * sizeof(float));
	 normalBufferData.fill('\0');
	 auto *rawNormalArray = reinterpret_cast<float*>(normalBufferData.data());

	 for(int i=0;i< nbindice;i+=3)
	 {
		QVector3D p1(rawVertexArray[indices[i]*3],rawVertexArray[indices[i]*3  +1],rawVertexArray[indices[i]*3+2]);
		 QVector3D p2(rawVertexArray[indices[i+1]*3],rawVertexArray[indices[i+1]*3+1],rawVertexArray[indices[i+1]*3+2]);
		 QVector3D p3(rawVertexArray[indices[i+2]*3],rawVertexArray[indices[i+2]*3+1],rawVertexArray[indices[i+2]*3+2]);

		 QVector3D v1 = p3 - p2;
		 QVector3D v2 = p1 - p2;

		 QVector3D n1 = QVector3D::crossProduct(v1,v2);
		n1 = n1.normalized();

		rawNormalArray[indices[i]*3]+= n1.x();
		rawNormalArray[indices[i]*3+1]+= n1.y();
		rawNormalArray[indices[i]*3+2]+= n1.z();
		rawNormalArray[indices[i+1]*3]+= n1.x();
		rawNormalArray[indices[i+1]*3+1]+= n1.y();
		rawNormalArray[indices[i+1]*3+2]+= n1.z();
		rawNormalArray[indices[i+2]*3]+= n1.x();
		rawNormalArray[indices[i+2]*3+1]+= n1.y();
		rawNormalArray[indices[i+2]*3+2]+= n1.z();

	 }
	 m_normalBuffer->setData(normalBufferData);
}

void SurfaceMesh::simplifyMesh(int target_count, double agressiveness)
{
	qDebug()<<" Simplify mesh , target count :"<<target_count;
	// init
	//loopi(0,triangles.size())
	for(int i=0;i<<triangles.size();i++)
	{
		triangles[i].deleted=0;
	}

	// main iteration loop
	int deleted_triangles=0;
	QVector<int> deleted0,deleted1;
	int triangle_count=triangles.size();
	//int iteration = 0;
	//loop(iteration,0,100)
	for (int iteration = 0; iteration < 100; iteration ++)
	{
		if(triangle_count-deleted_triangles<=target_count)break;

		// update mesh once in a while
		if(iteration%5==0)
		{
			update_mesh(iteration);
		}

		// clear dirty flag
		//loopi(0,triangles.size())
		for(int i=0;i<triangles.size();i++)
		{
		triangles[i].dirty=0;
		}

		//
		// All triangles with edges below the threshold will be removed
		//
		// The following numbers works well for most models.
		// If it does not, try to adjust the 3 parameters
		//
		double threshold = 0.000000001*pow(double(iteration+3),agressiveness);

		// target number of triangles reached ? Then break
		/*if ((verbose) && (iteration%5==0)) {
			printf("iteration %d - triangles %d threshold %g\n",iteration,triangle_count-deleted_triangles, threshold);
		}*/

		// remove vertices & mark deleted triangles
		//loopi(0,triangles.size())
		for(int i=0;i<triangles.size();i++)
		{
			Triangle &t=triangles[i];
			if(t.err[3]>threshold) continue;
			if(t.deleted) continue;
			if(t.dirty) continue;

			//loopj(0,3)
			for(int j=0;j<3;j++)
			if(t.err[j]<threshold)
			{

				int i0=t.v[ j     ]; Vertex &v0 = vertices[i0];
				int i1=t.v[(j+1)%3]; Vertex &v1 = vertices[i1];
				// Border check
				if(v0.border != v1.border)  continue;

				// Compute vertex to collapse to
				QVector3D p;
				calculate_error(i0,i1,p);
				deleted0.resize(v0.tcount); // normals temporarily
				deleted1.resize(v1.tcount); // normals temporarily
				// don't remove if flipped
				if( flipped(p,i0,i1,v0,v1,deleted0) ) continue;

				if( flipped(p,i1,i0,v1,v0,deleted1) ) continue;

				if ( (t.attr & TEXCOORD) == TEXCOORD  )
				{
					update_uvs(i0,v0,p,deleted0);
					update_uvs(i0,v1,p,deleted1);
				}

				// not flipped, so remove edge
				v0.p=p;
				v0.q=v1.q+v0.q;
				int tstart=refs.size();

				update_triangles(i0,v0,deleted0,deleted_triangles);
				update_triangles(i0,v1,deleted1,deleted_triangles);

				int tcount=refs.size()-tstart;

				if(tcount<=v0.tcount)
				{
					// save ram
					if(tcount)memcpy(&refs[v0.tstart],&refs[tstart],tcount*sizeof(Ref));
				}
				else
					// append
					v0.tstart=tstart;

				v0.tcount=tcount;
				break;
			}
			// done?
			if(triangle_count-deleted_triangles<=target_count)break;
		}
	}
	// clean up mesh
	compact_mesh();



}

void SurfaceMesh::update_mesh(int iteration)
{
	if(iteration>0) // compact triangles
	{
		int dst=0;
		//loopi(0,triangles.size())
		for(int i=0;i<triangles.size();i++)
		if(!triangles[i].deleted)
		{
			triangles[dst++]=triangles[i];
		}
		triangles.resize(dst);
	}
	//
	// Init Quadrics by Plane & Edge Errors
	//
	// required at the beginning ( iteration == 0 )
	// recomputing during the simplification is not required,
	// but mostly improves the result for closed meshes
	//
	if( iteration == 0 )
	{
		//loopi(0,vertices.size())
		for(int i=0;i<vertices.size();i++)
		vertices[i].q=SymetricMatrix(0.0);

		//loopi(0,triangles.size())
		for(int i=0;i<triangles.size();i++)
		{
			Triangle &t=triangles[i];
			QVector3D n,p[3];
			//loopj(0,3)
			for(int j=0;j<3;j++)
				p[j]=vertices[t.v[j]].p;
			n = QVector3D::crossProduct(p[1]-p[0],p[2]-p[0]); //n.cross(p[1]-p[0],p[2]-p[0]);
			n.normalize();
			t.n=n;
			//loopj(0,3)
			for(int j=0;j<3;j++)
			{
				vertices[t.v[j]].q =vertices[t.v[j]].q+SymetricMatrix(n.x(),n.y(),n.z(),QVector3D::dotProduct(-n,p[0]));  //-n.dot(p[0]));
			}

			/*loopj(0,3) vertices[t.v[j]].q =
					vertices[t.v[j]].q+SymetricMatrix(n.x,n.y,n.z,-n.dot(p[0]));
					*/
			//vertices[t.v[j]].q =vertices[t.v[j]].q+SymetricMatrix(n.x,n.y,n.z,QVector3D::dotProduct(-n,p[0]));  //-n.dot(p[0]));
		}
		//loopi(0,triangles.size())
		for(int i=0;i<triangles.size();i++)
		{
			// Calc Edge Error
			Triangle &t=triangles[i];QVector3D p;
			//loopj(0,3)
			for(int j=0;j<3;j++)
				t.err[j]=calculate_error(t.v[j],t.v[(j+1)%3],p);
			t.err[3]=fmin(t.err[0],fmin(t.err[1],t.err[2]));
		}
	}

	// Init Reference ID list
	//loopi(0,vertices.size())
	for(int i=0;i<vertices.size();i++)
	{
		vertices[i].tstart=0;
		vertices[i].tcount=0;
	}
	//loopi(0,triangles.size())
	for(int i=0;i<triangles.size();i++)
	{
		Triangle &t=triangles[i];
		//loopj(0,3)
		for(int j=0;j<3;j++)
			vertices[t.v[j]].tcount++;
	}
	int tstart=0;
	//loopi(0,vertices.size())
	for(int i=0;i<vertices.size();i++)
	{
		Vertex &v=vertices[i];
		v.tstart=tstart;
		tstart+=v.tcount;
		v.tcount=0;
	}

	// Write References
	refs.resize(triangles.size()*3);
	//loopi(0,triangles.size())
	for(int i=0;i<triangles.size();i++)
	{
		Triangle &t=triangles[i];
		//loopj(0,3)
		for(int j=0;j<3;j++)
		{
			Vertex &v=vertices[t.v[j]];
			refs[v.tstart+v.tcount].tid=i;
			refs[v.tstart+v.tcount].tvertex=j;
			v.tcount++;
		}
	}

	// Identify boundary : vertices[].border=0,1
	if( iteration == 0 )
	{
		QVector<int> vcount,vids;

		//loopi(0,vertices.size())
		for(int i=0;i<vertices.size();i++)
			vertices[i].border=0;

		//loopi(0,vertices.size())
		for(int i=0;i<vertices.size();i++)
		{
			Vertex &v=vertices[i];
			vcount.clear();
			vids.clear();
			//loopj(0,v.tcount)
			for(int j=0;j<v.tcount;j++)
			{
				int k=refs[v.tstart+j].tid;
				Triangle &t=triangles[k];
				//loopk(0,3)
				for(int k=0;k<3;k++)
				{
					int ofs=0,id=t.v[k];
					while(ofs<vcount.size())
					{
						if(vids[ofs]==id)break;
						ofs++;
					}
					if(ofs==vcount.size())
					{
						vcount.push_back(1);
						vids.push_back(id);
					}
					else
						vcount[ofs]++;
				}
			}
			//loopj(0,vcount.size())
			for(int j=0;j<vcount.size();j++)
				if(vcount[j]==1)
					vertices[vids[j]].border=1;
		}
	}
}

double SurfaceMesh::vertex_error(SymetricMatrix q, double x, double y, double z)
{
	return   q[0]*x*x + 2*q[1]*x*y + 2*q[2]*x*z + 2*q[3]*x + q[4]*y*y
		 + 2*q[5]*y*z + 2*q[6]*y + q[7]*z*z + 2*q[8]*z + q[9];
}

double SurfaceMesh::calculate_error(int id_v1, int id_v2, QVector3D &p_result)
{
	// compute interpolated vertex

	SymetricMatrix q = vertices[id_v1].q + vertices[id_v2].q;
	bool   border = vertices[id_v1].border & vertices[id_v2].border;
	double error=0;
	double det = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7);
	if ( det != 0 && !border )
	{

		// q_delta is invertible
		p_result.setX( -1/det*(q.det(1, 2, 3, 4, 5, 6, 5, 7 , 8)));	// vx = A41/det(q_delta)
		p_result.setY( 1/det*(q.det(0, 2, 3, 1, 5, 6, 2, 7 , 8)));	// vy = A42/det(q_delta)
		p_result.setZ( -1/det*(q.det(0, 1, 3, 1, 4, 6, 2, 5,  8)));	// vz = A43/det(q_delta)

		error = vertex_error(q, p_result.x(), p_result.y(), p_result.z());
	}
	else
	{
		// det = 0 -> try to find best result
		QVector3D p1=vertices[id_v1].p;
		QVector3D p2=vertices[id_v2].p;
		QVector3D p3=(p1+p2)/2;
		double error1 = vertex_error(q, p1.x(),p1.y(),p1.z());
		double error2 = vertex_error(q, p2.x(),p2.y(),p2.z());
		double error3 = vertex_error(q, p3.x(),p3.y(),p3.z());
		error = fmin(error1, fmin(error2, error3));
		if (error1 == error) p_result=p1;
		if (error2 == error) p_result=p2;
		if (error3 == error) p_result=p3;
	}
	return error;
}

bool SurfaceMesh::flipped(QVector3D p,int i0,int i1,Vertex &v0,Vertex &v1,QVector<int> &deleted)
{
	//loopk(0,v0.tcount)
	for(int k=0;k<v0.tcount;k++)
	{
		Triangle &t=triangles[refs[v0.tstart+k].tid];
		if(t.deleted)continue;

		int s=refs[v0.tstart+k].tvertex;
		int id1=t.v[(s+1)%3];
		int id2=t.v[(s+2)%3];

		if(id1==i1 || id2==i1) // delete ?
		{

			deleted[k]=1;
			continue;
		}
		QVector3D d1 = vertices[id1].p-p; d1.normalize();
		QVector3D d2 = vertices[id2].p-p; d2.normalize();
		float dot = QVector3D::dotProduct(d1,d2);
		if(fabs(dot)>0.999) return true;
		QVector3D n;
		n = QVector3D::crossProduct(d1,d2); //n.cross(d1,d2);
		n.normalize();
		deleted[k]=0;
		float dot2 = QVector3D::dotProduct(n,t.n);
		if(dot2<0.2) return true;
	}
	return false;
}

void SurfaceMesh::update_uvs(int i0,const Vertex &v,const QVector3D &p,QVector<int> &deleted)
{
	//loopk(0,v.tcount)

	for(int k=0;k<v.tcount;k++)
	{
		Ref &r=refs[v.tstart+k];
		Triangle &t=triangles[r.tid];
		if(t.deleted)continue;
		if(deleted[k])continue;
		QVector3D p1=vertices[t.v[0]].p;
		QVector3D p2=vertices[t.v[1]].p;
		QVector3D p3=vertices[t.v[2]].p;
		t.uvs[r.tvertex] = interpolate(p,p1,p2,p3,t.uvs);
	}
}

void SurfaceMesh::update_triangles(int i0,Vertex  &v,QVector<int> &deleted,int &deleted_triangles)
{
	QVector3D p;
	//loopk(0,v.tcount)
	for(int k=0;k<v.tcount;k++)
	{
		Ref &r=refs[v.tstart+k];
		Triangle &t=triangles[r.tid];
		if(t.deleted)continue;
		if(deleted[k])
		{
			t.deleted=1;
			deleted_triangles++;
			continue;
		}
		t.v[r.tvertex]=i0;
		t.dirty=1;
		t.err[0]=calculate_error(t.v[0],t.v[1],p);
		t.err[1]=calculate_error(t.v[1],t.v[2],p);
		t.err[2]=calculate_error(t.v[2],t.v[0],p);
		t.err[3]=fmin(t.err[0],fmin(t.err[1],t.err[2]));
		refs.push_back(r);
	}
}

void SurfaceMesh::compact_mesh()
{
	int dst=0;
	//loopi(0,vertices.size())
	for(int i=0;i<vertices.size();i++)
	{
		vertices[i].tcount=0;
	}
	//loopi(0,triangles.size())
	for(int i=0;i<triangles.size();i++)
	if(!triangles[i].deleted)
	{
		Triangle &t=triangles[i];
		triangles[dst++]=t;
		//loopj(0,3)
		for(int j=0;j<3;j++)
			vertices[t.v[j]].tcount=1;
	}
	triangles.resize(dst);
	dst=0;
	//loopi(0,vertices.size())
	for(int i=0;i<vertices.size();i++)
	if(vertices[i].tcount)
	{
		vertices[i].tstart=dst;
		vertices[dst].p=vertices[i].p;
		dst++;
	}
	//loopi(0,triangles.size())
	for(int i=0;i<triangles.size();i++)
	{
		Triangle &t=triangles[i];
		//loopj(0,3)
		for(int j=0;j<3;j++)
			t.v[j]=vertices[t.v[j]].tstart;
	}
	vertices.resize(dst);
}

QVector3D SurfaceMesh::barycentric(const QVector3D &p, const QVector3D &a, const QVector3D &b, const QVector3D &c)
{
	QVector3D v0 = b-a;
	QVector3D v1 = c-a;
	QVector3D v2 = p-a;
	double d00 =  QVector3D::dotProduct(v0,v0);//v0.dot(v0);
	double d01 = QVector3D::dotProduct(v0,v1);//v0.dot(v1);
	double d11 = QVector3D::dotProduct(v1,v1);//v1.dot(v1);
	double d20 = QVector3D::dotProduct(v2,v0);//v2.dot(v0);
	double d21 = QVector3D::dotProduct(v2,v1);//v2.dot(v1);
	double denom = d00*d11-d01*d01;
	double v = (d11 * d20 - d01 * d21) / denom;
	double w = (d00 * d21 - d01 * d20) / denom;
	double u = 1.0 - v - w;
	return QVector3D(u,v,w);
}

QVector2D SurfaceMesh::interpolate(const QVector3D &p, const QVector3D &a, const QVector3D &b, const QVector3D &c, const QVector2D attrs[3])
{
	QVector3D bary = barycentric(p,a,b,c);
	QVector2D out = QVector2D(0,0);
	out = out + attrs[0] * bary.x();
	out = out + attrs[1] * bary.y();
	//out = out + attrs[2] * bary.z();
	return out;
}

void SurfaceMesh::reaffectuv()
{
	for(int i=0;i<triangles.size();i++)
	{

		int indexVertex1 = triangles[i].v[0];
		vertices[indexVertex1].n += triangles[i].n;
		vertices[indexVertex1].uvs.setX(triangles[i].uvs[0].x());
		vertices[indexVertex1].uvs.setY(triangles[i].uvs[0].y());

		int indexVertex2 = triangles[i].v[1];
		vertices[indexVertex2].n += triangles[i].n;
		vertices[indexVertex2].uvs.setX(triangles[i].uvs[1].x());
		vertices[indexVertex2].uvs.setY(triangles[i].uvs[1].y());

		int indexVertex3 = triangles[i].v[2];
		vertices[indexVertex3].n += triangles[i].n;
		vertices[indexVertex3].uvs.setX(triangles[i].uvs[2].x());
		vertices[indexVertex3].uvs.setY(triangles[i].uvs[2].y());
	}

	for(int i=0;i<vertices.size();i++)
	{
		vertices[i].n =vertices[i].n.normalized();
	}
}

void SurfaceMesh::reloadFromCache(SurfaceMeshCache& cache,QMatrix4x4 sceneTransform)
{

	/*QByteArray vertexBufferDat(cache.vertexArray);
	//vertexBufferData.resize(3*cache.nbvertex * sizeof(float));
	//rawVertexArray = reinterpret_cast<float*>(vertexBufferData.data());

	QByteArray texBufferData;
	texBufferData.resize(2*cache.nbvertex * sizeof(float));
	rawTexArray = reinterpret_cast<float*>(texBufferData.data());

	QByteArray normalBufferData;
	normalBufferData.resize(3*cache.nbvertex * sizeof(float));
	rawNormalArray = reinterpret_cast<float*>(normalBufferData.data());

	//Indexes
	QByteArray indexBufferData;
	indexBufferData.resize(cache.nbTri*3 * sizeof(uint));
	rawIndexArray = reinterpret_cast<uint*>(indexBufferData.data());


*/

//	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	QByteArray vertexArraycopy(cache.vertexArray);
	float* rawVertexArray = reinterpret_cast<float*>(vertexArraycopy.data());

	for(int i=0;i<cache.nbVertex*3;i+=3)
	{
		QVector3D posi(rawVertexArray[i],rawVertexArray[i+1],rawVertexArray[i+2]);
		QVector3D posTr = sceneTransform * posi;
		rawVertexArray[i] = posTr.x();
		rawVertexArray[i+1] = posTr.y();
		rawVertexArray[i+2] = posTr.z();
	}

	m_vertexBuffer->setData(vertexArraycopy);
	m_texBuffer->setData(cache.textureArray);
	m_normalBuffer->setData(cache.normalArray);

	m_positionAttribute->setCount(cache.nbVertex);
	m_textureAttribute->setCount(cache.nbVertex);
	m_normalAttribute->setCount(cache.nbVertex);


	m_indexBuffer->setData(cache.indexArray);

	m_indexAttribute->setCount(3*cache.nbTri);

	setVertexCount(cache.nbTri*3);
	//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	//qDebug() << "Reload : " << std::chrono::duration<double, std::milli>(end-start).count();
}

template<typename IsoType>
struct SetupVertexKernel {
	static void run(IImagePaletteHolder* palette, int meshWidth, int meshHeight, int textureWidth, int textureHeight,
			float cubeOrigin, float cubeScale, int steps, QMatrix4x4 objectTransform, QVector<Vertex>& vertices,
			float nullValue, bool nullValueActive, float internalNullValue) {
		palette->lockPointer();
		IsoType* tab = static_cast<IsoType*>(palette->backingPointer());

		for (int row = 0; row < meshWidth; row+=1) {
			for (int col = 0; col < meshHeight; col+=1) {

				int bufferIndexMesh = col * meshWidth + row;

				int textureIndexWidth = ((float) row / (float ) meshWidth) * (float) textureWidth;
				textureIndexWidth = fmin(textureIndexWidth, textureWidth - 1);
				int textureIndexHeight = ((float) col / (float ) meshHeight) * (float) textureHeight;
				textureIndexHeight = fmin(textureIndexHeight, textureHeight - 1);

			//	int textureIndexWidth = row;
			//	int textureIndexHeight = col;

				double heightValue = tab[textureIndexWidth + textureIndexHeight * textureWidth];

				bool nullValueFound = nullValueActive && heightValue==nullValue;

				float newHeightValue =  cubeOrigin + cubeScale *heightValue;
				QVector3D inVect(row*steps, newHeightValue, col*steps);
				QVector3D outVect = objectTransform * inVect;

				if (nullValueFound) {
					outVect.setY(internalNullValue);
				}


				Vertex vert;
				vert.p = outVect;
				vert.n = QVector3D(0.0,-1.0,0.0);

				//QVector2D uv( col*1.0/(meshHeight-1),row*1.0/(meshWidth-1));
				float uvx =  row*1.0/((float)(textureWidth / (float)steps)-1);
				float uvy = col*1.0/((float)(textureHeight / (float)steps)-1);
				QVector2D uv(uvx, uvy);
				vert.uvs = uv;
				//uvs.push_back(uv);
				vertices.push_back(vert);

			}
		}

		palette->unlockPointer();
	}
};

void SurfaceMesh::init_obj(int width, int height,IImagePaletteHolder* palette, float heightThreshold,float cubeOrigin,float cubeScale, QString  path,int simplifySteps, int compression,QMatrix4x4 sceneTransform ,QMatrix4x4 objectTransform)
{
	//auto start = std::chrono::steady_clock::now();
	//qDebug()<<"init_obj !!!!!!!!!! "<<path;
	m_sceneTransform = sceneTransform;
	QFileInfo fileInfo(path);
	vertices.clear();
	triangles.clear();

	m_steps = simplifySteps;
	setDimensions(QVector2D(width, height));
	//auto init1 = std::chrono::steady_clock::now();
	if(path =="")
	{

		initMesh(width, height,palette, heightThreshold,cubeOrigin,cubeScale, simplifySteps);
		return;
	}


	//auto init2 = std::chrono::steady_clock::now();
	if( !fileInfo.exists())
	{

		//SurfaceGenerator surf(m_dimensions.x()/s,m_dimensions.y()/s, transform);
		int numVertex=(width) * (height);//  surf.getVerticesCount();

		int textureWidth = width;
		int textureHeight = height;

		int meshWidth = getWidth();
		 int meshHeight =getHeight();


		SampleTypeBinder binder(palette->sampleType());
		binder.bind<SetupVertexKernel>(palette, meshWidth, meshHeight, textureWidth, textureHeight,
				cubeOrigin, cubeScale, m_steps, objectTransform, vertices, m_nullValue, m_nullValueActive,
				m_internalNullValue);


/*	int sensIndex;
		const float*transfo = m_transform.constData();
		float xscale = transfo[0];
		float zscale = transfo[10];

		if((xscale > 0 && zscale > 0) || (xscale < 0 && zscale < 0))
		{
			sensIndex = -1;
		}else{
			sensIndex = 1;
		}*/

	int sensIndex  = getSensForIndex(width,height);

	//	qDebug()<<" init obj , sens index: "<<sensIndex<<" ,  xscale :"<<xscale<<" , zscale : "<<zscale;
		//qDebug()<<"init obj sens index : "<<sensIndex;



	for (int row = 0; row < meshWidth-1; row+=1) {
		for (int col = 0; col < meshHeight-1; col+=1) {
			//long i = (col + row * (meshHeight-1)) * 6;
			Triangle t1;

			t1.v[0] = col + row * meshHeight;
			t1.v[1] = col + 1 + row * meshHeight;
			t1.v[2] = col + (row + 1) * meshHeight;

			t1.uvs[0] =  vertices[t1.v[0]].uvs; //uvs[t1.v[0]];
			t1.uvs[1] =  vertices[t1.v[1]].uvs; //uvs[t1.v[1]];
			t1.uvs[2] =  vertices[t1.v[2]].uvs; // uvs[t1.v[2]];

			//QVector3D normal = QVector3D::crossProduct(p[1]-p[0],p[2]-p[0]); //n.cross(p[1]-p[0],p[2]-p[0]);
			QVector3D normal = QVector3D::crossProduct(vertices[t1.v[1]].p-vertices[t1.v[0]].p,vertices[t1.v[2]].p- vertices[t1.v[0]].p);
			normal.normalize();
			t1.n = -sensIndex *normal;//QVector3D(0.0,1.0,0.0);
			normalsVec.push_back(t1.n);
			triangles.push_back(t1);

			Triangle t2;
			t2.v[0] = col + (row + 1) * meshHeight;
			t2.v[1] = col + 1 + row * meshHeight;
			t2.v[2] = col + 1 + (row + 1) * meshHeight;

			t2.uvs[0] =  vertices[t2.v[0]].uvs;
			t2.uvs[1] =  vertices[t2.v[1]].uvs;
			t2.uvs[2] =  vertices[t2.v[2]].uvs;

			QVector3D normal2 = QVector3D::crossProduct(vertices[t2.v[1]].p-vertices[t2.v[0]].p,vertices[t2.v[2]].p- vertices[t2.v[0]].p);
			normal2.normalize();
			t2.n = -sensIndex *normal2 ;//QVector3D(0.0,1.0,0.0);

			normalsVec.push_back(t2.n);
			triangles.push_back(t2);

		}
	}





	if(compression > 0)
	{
		int nbsommet = vertices.size();
		int newNbSommet = (100- compression)*0.01f *nbsommet;
		simplifyMesh( newNbSommet,true);
	}

	reaffectuv();
	write_obj(path.toStdString().c_str());

	}
	//auto init3 = std::chrono::steady_clock::now();
	load_obj2(path.toStdString().c_str(),true);
//	auto end = std::chrono::steady_clock::now();

	/*qDebug() << "All : " << std::chrono::duration<double, std::milli>(end-start).count() <<
			", load : " << std::chrono::duration<double, std::milli>(end-init3).count() <<
			", other : " << std::chrono::duration<double, std::milli>(init3-start).count();*/
}


int SurfaceMesh::getSensForIndex(int width,int height)
{
//	qDebug()<<" matrx : "<<m_transform;
	//const float*transfo = m_transform.constData();
	QVector3D pos1 ( 0.0f,0.0f,0.0f);
		QVector3D pos2 (1.0,0.0f,0.0f);
		QVector3D pos3 ( 0.0f,0.0f,1.0);



		pos1 =  m_transform * pos1;
		pos2 = m_transform * pos2;
		pos3 =  m_transform * pos3;

		QVector3D dir1 = pos2 - pos1;
		QVector3D dir2 = pos3 - pos1;


		QVector3D normal = QVector3D::crossProduct(dir1,dir2);
		normal = normal.normalized();

		int sensIndex = 1;
		if( normal.y() > 0.0f)sensIndex =-1;

		return sensIndex;
}

template<typename InputType>
struct InitMeshTriangleSetupKernel {
	static void run(IImagePaletteHolder* palette, int width, int height, int steps, float cubeOrigin,
			float cubeScale, float* rawVertexArray, float* rawNormalArray, float* rawTexArray, QMatrix4x4& transform,
			float nullValue, bool nullValueActive, float internalNullValue) {
		palette->lockPointer();
		InputType* tab = static_cast<InputType*>(palette->backingPointer());

		int textureWidth = width;
		int textureHeight = height;


		int indexVertex=0;
		int indexUV=0;



	/*	float min = 100000.0f;
		float max = -100000.0f;
		float moyenne =0.0f;
		for (int row = 0; row < width; row+=1)
		{
			for (int col = 0; col < height; col+=1)
			{
				double heightValue =tab[row + col * width];
				float newHeightValue =  cubeOrigin + cubeScale *heightValue;
				if(newHeightValue <min ) min = newHeightValue;
				if(newHeightValue> max) max = newHeightValue;
				moyenne+= newHeightValue;
			}
		}


		moyenne = moyenne /(height*width);
		qDebug()<<" moyenne :"<<moyenne<<" , min :"<<min<<" , max :"<<max;
	*/
		/*for (int row = 0; row < width; row+=1)
		{
			for (int col = 0; col < height; col+=1)
			{
				QVector<double> voisins;
				double heightValue =tab[row + col * width]; voisins.push_back(heightValue);

				if( row> 0 && col > 0) {double heightValueHG =tab[row-1 + (col-1) * width];voisins.push_back(heightValueHG);}
				if( col > 0) {double heightValueH =tab[row + (col-1) * width];voisins.push_back(heightValueH);}
				if( row<width-1 && col > 0) {double heightValueHD =tab[row+1 + (col-1) * width];voisins.push_back(heightValueHD);}
				if( row> 0 && col > 0) {double heightValueG =tab[row -1+ col * width];voisins.push_back(heightValueG);}
				if( row<width-1 && col > 0) {double heightValueD =tab[row +1+ col * width];voisins.push_back(heightValueD);}
				if( row> 0 && col < height-1) {double heightValueBG =tab[row -1+ (col+1) * width];voisins.push_back(heightValueBG);}
				if( row> 0 && col < height-1) {double heightValueB =tab[row + (col+1) * width];voisins.push_back(heightValueB);}
				if( row<width-1 && col < height-1) {double heightValueBD =tab[row +1+ (col+1) * width];voisins.push_back(heightValueBD);}

				//trier
				qSort(voisins);


				//median
				int indexMedian = voisins.length() /2;
				tab[row + col * width] = voisins[indexMedian];


			}
		}*/

		for (int row = 0; row < width; row+=steps)
		{
			for (int col = 0; col < height; col+=steps)
			{

				double heightValue =tab[row + col * width];


				float newHeightValue =  cubeOrigin + cubeScale *heightValue;

				bool nullValueFound = nullValueActive && newHeightValue==nullValue;

				QVector3D inVect(row, newHeightValue, col);
				QVector3D outVect = transform * inVect;

				if (nullValueFound)
				{
					outVect.setY(internalNullValue);
				}

				rawVertexArray[indexVertex] = outVect.x();
				rawVertexArray[indexVertex+1] = outVect.y();
				rawVertexArray[indexVertex+2] = outVect.z();

				rawNormalArray[indexVertex]= 0.0f;
				rawNormalArray[indexVertex+1]= 0.0f;
				rawNormalArray[indexVertex+2]= 0.0f;

				rawTexArray[indexUV]=row*1.0/(width-1);
				rawTexArray[indexUV+1]=col*1.0/(height-1);

				indexVertex+=3;
				indexUV+=2;
			}
		}

	/*	for (int row = 0; row < meshWidth; row+=1) {
			for (int col = 0; col < meshHeight; col+=1) {

				int bufferIndexMesh = col * meshWidth + row;

				int textureIndexWidth = ((float) row / (float ) meshWidth) * (float) textureWidth;
				textureIndexWidth = fmin(textureIndexWidth, textureWidth - 1);
				int textureIndexHeight = ((float) col / (float ) meshHeight) * (float) textureHeight;
				textureIndexHeight = fmin(textureIndexHeight, textureHeight - 1);

				double heightValue = tab[textureIndexWidth + textureIndexHeight * textureWidth];

				float newHeightValue =  cubeOrigin + cubeScale *heightValue;
				QVector3D inVect(row*m_steps, newHeightValue, col*m_steps);
				QVector3D outVect = m_transform * inVect;

				rawVertexArray[indexVertex] = outVect.x();
				rawVertexArray[indexVertex+1] = outVect.y();
				rawVertexArray[indexVertex+2] = outVect.z();

				rawNormalArray[indexVertex]= 0.0f;
				rawNormalArray[indexVertex+1]= -1.0f;
				rawNormalArray[indexVertex+2]= 0.0f;

				float uvx = row*1.0/((float)(width / (float)m_steps) -1);
				float uvy = col*1.0/((float)(height/ (float)m_steps)-1);
				rawTexArray[indexUV]=uvx;
				rawTexArray[indexUV+1]=uvy;

				indexVertex+=3;
				indexUV+=2;


			}
		}*/



		palette->unlockPointer();
	}
};

void SurfaceMesh::initMesh(int width, int height,IImagePaletteHolder* palette, float heightThreshold,float cubeOrigin,float cubeScale,int simplifySteps)
{

	//qDebug()<<" init mesh";
	m_steps = simplifySteps;
	int meshWidth = (width-1) / m_steps + 1;
	int meshHeight =(height-1)  / m_steps + 1;
	int nbvertex=(meshWidth) * (meshHeight);
	int nbTri= (meshWidth-1) * (meshHeight-1)*2;

	QByteArray vertexBufferData;
	vertexBufferData.resize(3*nbvertex * sizeof(float));
	float* rawVertexArray = reinterpret_cast<float*>(vertexBufferData.data());

	QByteArray texBufferData;
	texBufferData.resize(2*nbvertex * sizeof(float));
	float* rawTexArray = reinterpret_cast<float*>(texBufferData.data());

	QByteArray normalBufferData;
	normalBufferData.resize(3*nbvertex * sizeof(float));
	float* rawNormalArray = reinterpret_cast<float*>(normalBufferData.data());

	//Indexes
	QByteArray indexBufferData;
	indexBufferData.resize(nbTri*3 * sizeof(uint));
	uint* rawIndexArray = reinterpret_cast<uint*>(indexBufferData.data());


	SampleTypeBinder binder(palette->sampleType());
	binder.bind<InitMeshTriangleSetupKernel>(palette, width, height, m_steps, cubeOrigin, cubeScale,
			rawVertexArray, rawNormalArray, rawTexArray, m_transform, m_nullValue, m_nullValueActive,
			m_internalNullValue);


	/*const float*transfo = m_transform.constData();
	float xscale = transfo[0];
	float zscale = transfo[10];
*/






	int sensIndex  = getSensForIndex(width,height);

	/*if((xscale > 0 && zscale > 0) || (xscale < 0 && zscale < 0))
	{
		sensIndex = 1;
	}else{
		sensIndex = -1;
	}*/



	//qDebug()<<" init mesh sens index : "<<sensIndex;
	if(sensIndex > 0)
		{
			#pragma omp parallel for
			for (int row = 0; row < meshWidth-1; row+=1) {
				for (int col = 0; col < meshHeight-1; col+=1) {
					long i = (col + row * (meshHeight-1)) * 6;

					rawIndexArray[i] = col + row * meshHeight;
					rawIndexArray[i+1] = col + 1 + row * meshHeight;
					rawIndexArray[i+2] = col + (row + 1) * meshHeight;


					rawIndexArray[i+3] = col + (row + 1) * meshHeight;
					rawIndexArray[i+4] = col + 1 + row * meshHeight;
					rawIndexArray[i+5] = col + 1 + (row + 1) * meshHeight;





				}
			}
		}
		else
		{
			#pragma omp parallel for
			for (int row = 0; row < meshWidth-1; row+=1) {
				for (int col = 0; col < meshHeight-1; col+=1) {
					long i = (col + row * (meshHeight-1)) * 6;
					rawIndexArray[i++] = col + row * meshHeight;
					rawIndexArray[i++] = col + (row + 1) * meshHeight;
					rawIndexArray[i++] = col + 1 + row * meshHeight;


					rawIndexArray[i++] = col + (row + 1) * meshHeight;
					rawIndexArray[i++] = col + 1 + (row + 1) * meshHeight;
					rawIndexArray[i++] = col + 1 + row * meshHeight;

				}
			}
		}



	QVector<MinMax> listeindex;


	for(int i=0;i<3*nbTri;i+=3)
	{
		QVector3D p1(rawVertexArray[rawIndexArray[i]*3],rawVertexArray[rawIndexArray[i]*3  +1],rawVertexArray[rawIndexArray[i]*3+2]);
		 QVector3D p2(rawVertexArray[rawIndexArray[i+1]*3],rawVertexArray[rawIndexArray[i+1]*3+1],rawVertexArray[rawIndexArray[i+1]*3+2]);
		 QVector3D p3(rawVertexArray[rawIndexArray[i+2]*3],rawVertexArray[rawIndexArray[i+2]*3+1],rawVertexArray[rawIndexArray[i+2]*3+2]);

		 QVector3D v1 = p3 - p1;  //3-2
		 QVector3D v2 = p2 - p1;  //1-2

		 v1 = v1.normalized();
		 v2 = v2.normalized();

		 QVector3D n1 = QVector3D::crossProduct(v1,v2);
		 n1 = n1.normalized();

		 filterTrianglePoints(i, p1, p2, p3, rawIndexArray, rawVertexArray, listeindex, n1);


		 for (int k=0; k<3; k++) {


			 rawNormalArray[rawIndexArray[i+k]*3] += n1.x();
			 rawNormalArray[rawIndexArray[i+k]*3+1] += n1.y();
			 rawNormalArray[rawIndexArray[i+k]*3+2] += n1.z();
		 }


		/* for (int k=0; k<3; k++) {
			 rawNormalArray[rawIndexArray[i+k]*3] += n1.x();
			 rawNormalArray[rawIndexArray[i+k]*3+1] += n1.y();
			 rawNormalArray[rawIndexArray[i+k]*3+2] += n1.z();
		 }*/


	}


	for (int i=0;i<nbvertex*3;i+=3) {
		QVector3D vect(rawNormalArray[i], rawNormalArray[i+1], rawNormalArray[i+2]);
		vect = vect.normalized();


		rawNormalArray[i] = vect.x();
		rawNormalArray[i+1] = vect.y();
		rawNormalArray[i+2] = vect.z();
	}


/*	m_listeNormals.clear();

	qDebug() <<" nbVertex:"<<nbvertex;
	for(int i=0;i<3*nbvertex;i+=3)
	   {

		QVector3D N1 = QVector3D(rawVertexArray[i], rawVertexArray[i+1], rawVertexArray[i+2]);
		QVector3D N2 = N1 + QVector3D(rawNormalArray[i], rawNormalArray[i+1], rawNormalArray[i+2])*20.0f;
		m_listeNormals.push_back(N1);
		m_listeNormals.push_back(N2);
	   }*/

	//Qt3DHelpers::drawNormals(m_listeNormals,Qt::red,this,1);

	//qDebug() <<" listeindex count :"<<listeindex.count();

	for( int i=0;i< listeindex.count();i++)
	{
		//float yy = rawVertexArray[rawIndexArray[i-1]+1];
	//	int val = rawVertexArray[listeindex[i].indice];
		rawVertexArray[listeindex[i].indice] = listeindex[i].valeur;
		//qDebug() <<val <<" <==> "<<rawVertexArray[listeindex[i].indice];
	}


	m_vertexBuffer->setData(vertexBufferData);
	m_texBuffer->setData(texBufferData);
	m_normalBuffer->setData(normalBufferData);

	m_positionAttribute->setCount(nbvertex);
	m_textureAttribute->setCount(nbvertex);
	m_normalAttribute->setCount(nbvertex);


	m_indexBuffer->setData(indexBufferData);

	m_indexAttribute->setCount(3*nbTri);

	setVertexCount(nbTri*3);

}





//Option : Load OBJ
void SurfaceMesh::load_obj(const char* filename, bool process_uv){
		vertices.clear();
		triangles.clear();
		//qDebug()<<" load obj "<<filename;
		//printf ( "Loading Objects %s ... \n",filename);
		FILE* fn;
		if(filename==NULL)		return ;
		if((char)filename[0]==0)	return ;
		if ((fn = fopen(filename, "rb")) == NULL)
		{
			printf ( "File %s not found!\n" ,filename );
			return;
		}
		char line[1000];
		memset ( line,0,1000 );
		int vertex_cnt = 0;
		int material = -1;
		//std::map<std::string, int> material_map;
		QVector<QVector2D> uvs;
		QVector<QVector<int> > uvMap;

		while(fgets( line, 1000, fn ) != NULL)
		{
			Vertex v;
			float vpx,vpy,vpz;
			//QVector3D uv;
			float uvx,uvy,uvz;

			/*if (strncmp(line, "mtllib", 6) == 0)
			{
				mtllib = trimwhitespace(&line[7]);
			}
			if (strncmp(line, "usemtl", 6) == 0)
			{
				std::string usemtl = trimwhitespace(&line[7]);
				if (material_map.find(usemtl) == material_map.end())
				{
					material_map[usemtl] = materials.size();
					materials.push_back(usemtl);
				}
				material = material_map[usemtl];
			}*/

			if ( line[0] == 'v' && line[1] == 't' )
			{
				if ( line[2] == ' ' )
				if(sscanf(line,"vt %lf %lf",
					&uvx,&uvy)==2)
				{
					uvz = 0;
					QVector2D uv(uvx,uvy);//,uvz);
					uvs.push_back(uv);
				} else
				if(sscanf(line,"vt %lf %lf %lf",
					&uvx,&uvy,&uvz)==3)
				{
					QVector2D uv(uvx,uvy);//,uvz);
					uvs.push_back(uv);
				}
			}
			else if ( line[0] == 'v' )
			{
				if ( line[1] == ' ' )
				if(sscanf(line,"v %lf %lf %lf",
					&vpx,	&vpy,	&vpz)==3)
				{
					Vertex v;
					v.p.setX(vpx);
					v.p.setY(vpy);
					v.p.setZ(vpz);
					vertices.push_back(v);
				}
			}
			int integers[9];
			if ( line[0] == 'f' )
			{
				Triangle t;
				bool tri_ok = false;
                bool has_uv = false;

				if(sscanf(line,"f %d %d %d",
					&integers[0],&integers[1],&integers[2])==3)
				{
					tri_ok = true;
				}else
				if(sscanf(line,"f %d// %d// %d//",
					&integers[0],&integers[1],&integers[2])==3)
				{
					tri_ok = true;
				}else
				if(sscanf(line,"f %d//%d %d//%d %d//%d",
					&integers[0],&integers[3],
					&integers[1],&integers[4],
					&integers[2],&integers[5])==6)
				{
					tri_ok = true;
				}else
				if(sscanf(line,"f %d/%d/%d %d/%d/%d %d/%d/%d",
					&integers[0],&integers[6],&integers[3],
					&integers[1],&integers[7],&integers[4],
					&integers[2],&integers[8],&integers[5])==9)
				{
					tri_ok = true;
					has_uv = true;
				}else // Add Support for v/vt only meshes
				if (sscanf(line, "f %d/%d %d/%d %d/%d",
					&integers[0], &integers[6],
					&integers[1], &integers[7],
					&integers[2], &integers[8]) == 6)
				{
					tri_ok = true;
					has_uv = true;
				}
				else
				{
					printf("unrecognized sequence\n");
					printf("%s\n",line);
					while(1);
				}
				if ( tri_ok )
				{
					t.v[0] = integers[0]-1-vertex_cnt;
					t.v[1] = integers[1]-1-vertex_cnt;
					t.v[2] = integers[2]-1-vertex_cnt;
					t.attr = 0;

					if ( process_uv && has_uv )
					{
						QVector<int> indices;
						indices.push_back(integers[6]-1-vertex_cnt);
						indices.push_back(integers[7]-1-vertex_cnt);
						indices.push_back(integers[8]-1-vertex_cnt);
						uvMap.push_back(indices);
						t.attr |= TEXCOORD;
					}

					//t.material = material;
					//geo.triangles.push_back ( tri );
					triangles.push_back(t);
					//state_before = state;
					//state ='f';
				}
			}
		}

		if ( process_uv && uvs.size() )
		{
			//loopi(0,triangles.size())
			for(int i=0;i<triangles.size();i++)
			{
				//loopj(0,3)
				for(int j=0;j<3;j++)
					triangles[i].uvs[j] = uvs[uvMap[i][j]];
			}
		}

		fclose(fn);

		//printf("load_obj: vertices = %lu, triangles = %lu, uvs = %lu\n", vertices.size(), triangles.size(), uvs.size() );
	} // load_obj()

void SurfaceMesh::loadObj(const char* filename, SurfaceMeshCache& cache)
{

			//printf ( "Loading Objects %s ... \n",filename);
			FILE* fn;
			if(filename==NULL)		return ;
			if((char)filename[0]==0)	return ;
			if ((fn = fopen(filename, "rb")) == NULL)
			{
				printf ( "File %s not found!\n" ,filename );
				return;
			}
			char line[1000];
			memset ( line,0,1000 );
			int vertex_cnt = 0;
			int material = -1;
			//std::map<std::string, int> material_map;
			QVector<QVector2D> uvs;
			QVector<QVector<int> > uvMap;

			int indexVert = 0;
			int indexUV=0;
			int indexN =0;
			int indexIndic =0;
			float* rawVertexArray;
			float* rawTexArray;
			float* rawNormalArray;
			uint* rawIndexArray;

			qDebug()<<" open obj2 : "<<filename;
			while(fgets( line, 1000, fn ) != NULL)
			{
				Vertex v;
				float vpx,vpy,vpz;
				float uvx,uvy,uvz;
				float nx,ny,nz;


				if ( line[0] == 'n' && line[1] == 'b' )
				{
					int nbvertex,nbTri;
					if( sscanf(line,"nb %d %d",&nbvertex,&nbTri)==2)
					{
						cache.nbVertex  = nbvertex;
						cache.nbTri = nbTri;

						//QByteArray vertexBufferData;
						cache.vertexArray.resize(3*nbvertex * sizeof(float));
						rawVertexArray = reinterpret_cast<float*>(cache.vertexArray.data());

						//QByteArray texBufferData;
						cache.textureArray.resize(2*nbvertex * sizeof(float));
						rawTexArray = reinterpret_cast<float*>(cache.textureArray.data());

						//QByteArray normalBufferData;
						cache.normalArray.resize(3*nbvertex * sizeof(float));
						rawNormalArray = reinterpret_cast<float*>(cache.normalArray.data());


						//Indexes
						//QByteArray indexBufferData;
						cache.indexArray.resize(nbTri*3 * sizeof(uint));
						rawIndexArray = reinterpret_cast<uint*>(cache.indexArray.data());

					}
				}

				if ( line[0] == 'v' && line[1] == 'n' )
				{
					if(sscanf(line,"vn %f %f %f",&nx,&ny,&nz)==3)
					{
						rawNormalArray[indexN++] = nx;
						rawNormalArray[indexN++] = ny;
						rawNormalArray[indexN++] = nz;
						//QVector3D uv(uvx,uvy,uvz);
						//uvs.push_back(uv);
					}
				}
				else if ( line[0] == 'v' && line[1] == 't' )
				{
					if ( line[2] == ' ' )
					if(sscanf(line,"vt %f %f",
						&uvx,&uvy)==2)
					{
						uvz = 0;
						rawTexArray[indexUV] = uvx;
						indexUV++;
						rawTexArray[indexUV] = uvy;
						indexUV++;
						//qDebug()<<indexUV <<" ==== uv :"<<uvx<<" ,"<<uvy;
						//QVector3D uv(uvx,uvy,uvz);
						//uvs.push_back(uv);
					} else
					if(sscanf(line,"vt %f %f %f",
						&uvx,&uvy,&uvz)==3)
					{
						rawTexArray[indexUV++] = uvx;
						rawTexArray[indexUV++] = uvy;
						//QVector3D uv(uvx,uvy,uvz);
						//uvs.push_back(uv);
					}
				}
				else if ( line[0] == 'v' )
				{
					if ( line[1] == ' ' )
					if(sscanf(line,"v %f %f %f",
						&vpx,	&vpy,	&vpz)==3)
					{

						QVector3D posi(vpx, vpy,vpz);
						QVector3D ptsTr =/*m_transform */ posi;
						rawVertexArray[indexVert]=ptsTr.x();
						indexVert++;
						rawVertexArray[indexVert]=ptsTr.y();
						indexVert++;
						rawVertexArray[indexVert]=ptsTr.z();
						indexVert++;

					}
				}
				int integers[9];
				if ( line[0] == 'f' )
				{
					//Triangle t;
					bool tri_ok = false;
	                bool has_uv = false;

					if(sscanf(line,"f %d %d %d",
						&integers[0],&integers[1],&integers[2])==3)
					{
						tri_ok = true;
					}else
					if(sscanf(line,"f %d// %d// %d//",
						&integers[0],&integers[1],&integers[2])==3)
					{
						tri_ok = true;
					}else
					if(sscanf(line,"f %d//%d %d//%d %d//%d",
						&integers[0],&integers[3],
						&integers[1],&integers[4],
						&integers[2],&integers[5])==6)
					{
						tri_ok = true;
					}else
					if(sscanf(line,"f %d/%d/%d %d/%d/%d %d/%d/%d",
						&integers[0],&integers[6],&integers[3],
						&integers[1],&integers[7],&integers[4],
						&integers[2],&integers[8],&integers[5])==9)
					{
						tri_ok = true;
						has_uv = true;
					}else // Add Support for v/vt only meshes
					if (sscanf(line, "f %d/%d %d/%d %d/%d",
						&integers[0], &integers[6],
						&integers[1], &integers[7],
						&integers[2], &integers[8]) == 6)
					{
						tri_ok = true;
						has_uv = true;
					}
					else
					{
						printf("unrecognized sequence\n");
						printf("%s\n",line);
						while(1);
					}
					if ( tri_ok )
					{

						rawIndexArray[indexIndic++] =1* (integers[0]-1-vertex_cnt);
						rawIndexArray[indexIndic++] =1* (integers[1]-1-vertex_cnt);
						rawIndexArray[indexIndic++] =1*( integers[2]-1-vertex_cnt);


						//t.attr = 0;

						if ( has_uv )
						{
							QVector<int> indices;
							indices.push_back(integers[6]-1-vertex_cnt);
							indices.push_back(integers[7]-1-vertex_cnt);
							indices.push_back(integers[8]-1-vertex_cnt);
							uvMap.push_back(indices);
							//t.attr |= TEXCOORD;
						}

						//t.material = material;
						//geo.triangles.push_back ( tri );
						//triangles.push_back(t);
						//state_before = state;
						//state ='f';
					}
				}
			}


			fclose(fn);
}

void SurfaceMesh::writeObj(const char* filename, QVector<Triangle> triangles,QVector<Vertex> vertices)
	{
		FILE *file=fopen(filename, "w");
		int cur_material = -1;
		bool has_uv = true;//(triangles.size() && (triangles[0].attr & TEXCOORD) == TEXCOORD);
		if (!file)
		{
			printf("write_obj: can't write data file \"%s\".\n", filename);
			exit(0);
		}
		/*if (!mtllib.empty()
		{
			fprintf(file, "mtllib %s\n", mtllib.c_str());
		}*/
		///loopi(0,vertices.size())
		//qDebug()<<"===> nb vertices : "<<vertices.size();
		fprintf(file, "nb %d %d \n",vertices.size(),triangles.size());

		for(int i=0;i<vertices.size();i++)
		{
			//fprintf(file, "v %lf %lf %lf\n", vertices[i].p.x,vertices[i].p.y,vertices[i].p.z);
			fprintf(file, "v %g %g %g\n", vertices[i].p.x(),vertices[i].p.y(),vertices[i].p.z()); //more compact: remove trailing zeros
		}

		for(int i=0;i<vertices.size();i++)
		{
			//fprintf(file, "v %lf %lf %lf\n", vertices[i].p.x,vertices[i].p.y,vertices[i].p.z);
			//qDebug()<<"vertices[i].uvs :"<<vertices[i].uvs;


			fprintf(file, "vt %g %g\n", vertices[i].uvs.x(),vertices[i].uvs.y()); //more compact: remove trailing zeros
		}

		for(int i=0;i<vertices.size();i++)
		{
			//fprintf(file, "v %lf %lf %lf\n", vertices[i].p.x,vertices[i].p.y,vertices[i].p.z);
			//qDebug()<<"vertices[i].uvs :"<<vertices[i].uvs;
			fprintf(file, "vn %g %g %g\n", vertices[i].n.x(),vertices[i].n.y(),vertices[i].n.z());
		}
		if (has_uv)
		{
			//loopi(0,triangles.size())
			/*for(int i=0;i<triangles.size();i++)
			if(!triangles[i].deleted)
			{
				fprintf(file, "vt %g %g\n", triangles[i].uvs[0].x(), triangles[i].uvs[0].y());
				fprintf(file, "vt %g %g\n", triangles[i].uvs[1].x(), triangles[i].uvs[1].y());
				fprintf(file, "vt %g %g\n", triangles[i].uvs[2].x(), triangles[i].uvs[2].y());
			}*/
		}
		int uv = 1;
		//loopi(0,triangles.size())

		for(int i=0;i<triangles.size();i++)
		{

			if(!triangles[i].deleted)
			{
			/*	if (triangles[i].material != cur_material)
				{
					cur_material = triangles[i].material;
					fprintf(file, "usemtl %s\n", materials[triangles[i].material].c_str());
				}*/
				if (has_uv)
				{
					 //if(i<120 )qDebug()<<" ==>"<<triangles[i].v[0]+1<<" ,  "<< triangles[i].v[1]+1<<" , "<< triangles[i].v[2]+1;
					fprintf(file, "f %d/%d %d/%d %d/%d\n", triangles[i].v[0]+1, triangles[i].v[0]+1, triangles[i].v[1]+1, triangles[i].v[1]+1, triangles[i].v[2]+1, triangles[i].v[2]+1);
					//uv += 3;
				}
				else
				{
				   // if(i<120 )qDebug()<< triangles[i].v[0]+1<<" ,  "<< triangles[i].v[1]+1<<" , "<< triangles[i].v[2]+1;
					fprintf(file, "f %d %d %d\n", triangles[i].v[0]+1, triangles[i].v[1]+1, triangles[i].v[2]+1);
				}
				//fprintf(file, "f %d// %d// %d//\n", triangles[i].v[0]+1, triangles[i].v[1]+1, triangles[i].v[2]+1); //more compact: remove trailing zeros
			}
		}
		fclose(file);
		qDebug()<<" fin du fichier";
	}


void SurfaceMesh::load_obj2(const char* filename, bool process_uv){ //,float *vertex,float * texCoord, float * normals, unsigned int* indices){
		//std::chrono::steady_clock::time_point startInit, endInit;
	//	auto start = std::chrono::steady_clock::now();
		vertices.clear();
		triangles.clear();

		//printf ( "Loading Objects %s ... \n",filename);
		FILE* fn;
		if(filename==NULL)		return ;
		if((char)filename[0]==0)	return ;
		if ((fn = fopen(filename, "rb")) == NULL)
		{
			printf ( "File %s not found!\n" ,filename );
			return;
		}
		char line[1000];
		memset ( line,0,1000 );
		int vertex_cnt = 0;
		int material = -1;
		//std::map<std::string, int> material_map;
		QVector<QVector3D> uvs;
		QVector<QVector<int> > uvMap;

		int indexVert = 0;
		int indexUV=0;
		int indexN =0;
		int indexIndic =0;
		float* rawVertexArray;
		float* rawTexArray;
		float* rawNormalArray;
		uint* rawIndexArray;

	//	qDebug()<<" open obj2 : "<<filename;
		while(fgets( line, 1000, fn ) != NULL)
		{
			Vertex v;
			float vpx,vpy,vpz;
			float uvx,uvy,uvz;
			float nx,ny,nz;


			/*if (strncmp(line, "mtllib", 6) == 0)
			{
				mtllib = trimwhitespace(&line[7]);
			}
			if (strncmp(line, "usemtl", 6) == 0)
			{
				std::string usemtl = trimwhitespace(&line[7]);
				if (material_map.find(usemtl) == material_map.end())
				{
					material_map[usemtl] = materials.size();
					materials.push_back(usemtl);
				}
				material = material_map[usemtl];
			}*/
			if ( line[0] == 'n' && line[1] == 'b' )
			{

				int nbvertex,nbTri;
				if( sscanf(line,"nb %d %d",&nbvertex,&nbTri)==2)
				{
					//qDebug()<<" read nb vertex :"<<nbvertex<<" save nb tri :"<<nbTri;
					//initGeometry(nbvertex,nbTri);
					QByteArray vertexBufferData;
					vertexBufferData.resize(3*nbvertex * sizeof(float));
					rawVertexArray = reinterpret_cast<float*>(vertexBufferData.data());

					QByteArray texBufferData;
					texBufferData.resize(2*nbvertex * sizeof(float));
					rawTexArray = reinterpret_cast<float*>(texBufferData.data());

					QByteArray normalBufferData;
					normalBufferData.resize(3*nbvertex * sizeof(float));
					rawNormalArray = reinterpret_cast<float*>(normalBufferData.data());

					//surf.getVertices(rawVertexArray,rawTexArray,rawNormalArray);


					//Indexes
					QByteArray indexBufferData;
					indexBufferData.resize(nbTri*3 * sizeof(uint));
					rawIndexArray = reinterpret_cast<uint*>(indexBufferData.data());
					//surf.getIndices(rawIndexArray);
//					uvMap.();
//					triangles.();


					m_vertexBuffer->setData(vertexBufferData);
					m_texBuffer->setData(texBufferData);
					m_normalBuffer->setData(normalBufferData);

					m_positionAttribute->setCount(nbvertex);
					m_textureAttribute->setCount(nbvertex);
					m_normalAttribute->setCount(nbvertex);


					m_indexBuffer->setData(indexBufferData);

					m_indexAttribute->setCount(3*nbTri);

					setVertexCount(nbTri*3);

				}

			}

			if ( line[0] == 'v' && line[1] == 'n' )
			{
				if(sscanf(line,"vn %f %f %f",&nx,&ny,&nz)==3)
				{
					rawNormalArray[indexN++] = nx;
					rawNormalArray[indexN++] = ny;
					rawNormalArray[indexN++] = nz;
					//QVector3D uv(uvx,uvy,uvz);
					//uvs.push_back(uv);
				}
			}
			else if ( line[0] == 'v' && line[1] == 't' )
			{
				if ( line[2] == ' ' )
				if(sscanf(line,"vt %f %f",
					&uvx,&uvy)==2)
				{
					uvz = 0;
					rawTexArray[indexUV] = uvx;
					indexUV++;
					rawTexArray[indexUV] = uvy;
					indexUV++;
					//qDebug()<<indexUV <<" ==== uv :"<<uvx<<" ,"<<uvy;
					//QVector3D uv(uvx,uvy,uvz);
					//uvs.push_back(uv);
				} else
				if(sscanf(line,"vt %f %f %f",
					&uvx,&uvy,&uvz)==3)
				{
					rawTexArray[indexUV++] = uvx;
					rawTexArray[indexUV++] = uvy;
					//QVector3D uv(uvx,uvy,uvz);
					//uvs.push_back(uv);
				}
			}
			else if ( line[0] == 'v' )
			{
				if ( line[1] == ' ' )
				if(sscanf(line,"v %f %f %f",
					&vpx,	&vpy,	&vpz)==3)
				{

					QVector3D posi(vpx, vpy,vpz);
					QVector3D ptsTr =m_sceneTransform * posi;
					rawVertexArray[indexVert]=ptsTr.x();
					indexVert++;
					rawVertexArray[indexVert]=ptsTr.y();
					indexVert++;
					rawVertexArray[indexVert]=ptsTr.z();
					indexVert++;


					//	qDebug()<<indexVert-2<<"  vertex "<<vertex[indexVert-3]<<" ,"<<vertex[indexVert-2]<<" , " <<vertex[indexVert-1];
					//Vertex v;
					//v.p.setX(vpx);
					//v.p.setY(vpy);
					//v.p.setZ(vpz);
					//vertices.push_back(v);

				}
			}
			int integers[9];
			if ( line[0] == 'f' )
			{
				Triangle t;
				bool tri_ok = false;
                bool has_uv = false;

				if(sscanf(line,"f %d %d %d",
					&integers[0],&integers[1],&integers[2])==3)
				{
					tri_ok = true;
				}else
				if(sscanf(line,"f %d// %d// %d//",
					&integers[0],&integers[1],&integers[2])==3)
				{
					tri_ok = true;
				}else
				if(sscanf(line,"f %d//%d %d//%d %d//%d",
					&integers[0],&integers[3],
					&integers[1],&integers[4],
					&integers[2],&integers[5])==6)
				{
					tri_ok = true;
				}else
				if(sscanf(line,"f %d/%d/%d %d/%d/%d %d/%d/%d",
					&integers[0],&integers[6],&integers[3],
					&integers[1],&integers[7],&integers[4],
					&integers[2],&integers[8],&integers[5])==9)
				{
					tri_ok = true;
					has_uv = true;
				}else // Add Support for v/vt only meshes
				if (sscanf(line, "f %d/%d %d/%d %d/%d",
					&integers[0], &integers[6],
					&integers[1], &integers[7],
					&integers[2], &integers[8]) == 6)
				{
					tri_ok = true;
					has_uv = true;
				}
				else
				{
					printf("unrecognized sequence\n");
					printf("%s\n",line);
					while(1);
				}
				if ( tri_ok )
				{
					//t.v[0] = integers[0]-1-vertex_cnt;
					//t.v[1] = integers[1]-1-vertex_cnt;
					//t.v[2] = integers[2]-1-vertex_cnt;
					//if(indexIndic<2095601)
					//{
					rawIndexArray[indexIndic++] =1* (integers[0]-1-vertex_cnt);
					rawIndexArray[indexIndic++] =1* (integers[1]-1-vertex_cnt);
					rawIndexArray[indexIndic++] =1*( integers[2]-1-vertex_cnt);

				//	if(integers[0])

					//	int indice1= 3*(integers[0]-1-vertex_cnt);
					//	int indice2= 3*(integers[1]-1-vertex_cnt);
					//	int indice3= 3*(integers[2]-1-vertex_cnt);


					/*	qDebug()<<indice1<<" ," <<indice2<<" ," <<indice3;
						qDebug()<<vertex[indice1]<<vertex[indice1+1]<<vertex[indice1+2];
						qDebug()<<vertex[indice2]<<vertex[indice2+1]<<vertex[indice2+2];
						qDebug()<<vertex[indice3]<<vertex[indice3+1]<<vertex[indice3+2];*/

	//				}

//
					t.attr = 0;

					if ( process_uv && has_uv )
					{
//						QVector<int> indices;
//						indices.push_back(integers[6]-1-vertex_cnt);
//						indices.push_back(integers[7]-1-vertex_cnt);
//						indices.push_back(integers[8]-1-vertex_cnt);
//						uvMap.push_back(indices);
						t.attr |= TEXCOORD;
					}

					//t.material = material;
					//geo.triangles.push_back ( tri );
					//triangles.push_back(t);
					//state_before = state;
					//state ='f';
				}
			}
		}

	/*	if ( process_uv && uvs.size() )
		{
			//loopi(0,triangles.size())
			for(int i=0;i<triangles.size();i++)
			{
				//loopj(0,3)
				for(int j=0;j<3;j++)
					triangles[i].uvs[j] = uvs[uvMap[i][j]];
			}
		}*/

		fclose(fn);

/*		auto end = std::chrono::steady_clock::now();
		qDebug() << "load obj All : " << std::chrono::duration<double, std::milli>(end-start).count() <<
				", Pre init : " << std::chrono::duration<double, std::milli>(startInit-start).count() <<
				", Init : " << std::chrono::duration<double, std::milli>(endInit-startInit).count() <<
				", Read : " << std::chrono::duration<double, std::milli>(end-endInit).count();*/
		//printf("load_obj: vertices = %lu, triangles = %lu, uvs = %lu\n", vertices.size(), triangles.size(), uvs.size() );
	} // load_obj()


void SurfaceMesh::write_obj(const char* filename)
	{
	qDebug()<<"write obj ";
		FILE *file=fopen(filename, "w");
		int cur_material = -1;
		bool has_uv = true;//(triangles.size() && (triangles[0].attr & TEXCOORD) == TEXCOORD);
		if (!file)
		{
			printf("write_obj: can't write data file \"%s\".\n", filename);
			exit(0);
		}
		/*if (!mtllib.empty())
		{
			fprintf(file, "mtllib %s\n", mtllib.c_str());
		}*/
		///loopi(0,vertices.size())
		//qDebug()<<"===> nb vertices : "<<vertices.size();
		fprintf(file, "nb %d %d \n",vertices.size(),triangles.size());

		for(int i=0;i<vertices.size();i++)
		{
			//fprintf(file, "v %lf %lf %lf\n", vertices[i].p.x,vertices[i].p.y,vertices[i].p.z);
			fprintf(file, "v %g %g %g\n", vertices[i].p.x(),vertices[i].p.y(),vertices[i].p.z()); //more compact: remove trailing zeros
		}

		for(int i=0;i<vertices.size();i++)
		{
			//fprintf(file, "v %lf %lf %lf\n", vertices[i].p.x,vertices[i].p.y,vertices[i].p.z);
			//qDebug()<<"vertices[i].uvs :"<<vertices[i].uvs;


			fprintf(file, "vt %g %g\n", vertices[i].uvs.x(),vertices[i].uvs.y()); //more compact: remove trailing zeros
		}

		for(int i=0;i<vertices.size();i++)
		{
			//fprintf(file, "v %lf %lf %lf\n", vertices[i].p.x,vertices[i].p.y,vertices[i].p.z);
			//qDebug()<<"vertices[i].uvs :"<<vertices[i].uvs;
			fprintf(file, "vn %g %g %g\n", vertices[i].n.x(),vertices[i].n.y(),vertices[i].n.z());
		}
		if (has_uv)
		{
			//loopi(0,triangles.size())
			/*for(int i=0;i<triangles.size();i++)
			if(!triangles[i].deleted)
			{
				fprintf(file, "vt %g %g\n", triangles[i].uvs[0].x(), triangles[i].uvs[0].y());
				fprintf(file, "vt %g %g\n", triangles[i].uvs[1].x(), triangles[i].uvs[1].y());
				fprintf(file, "vt %g %g\n", triangles[i].uvs[2].x(), triangles[i].uvs[2].y());
			}*/
		}
		int uv = 1;
		//loopi(0,triangles.size())
		//qDebug()<<" write triangle size : "<<triangles.size()*3;
		for(int i=0;i<triangles.size();i++)
		{

			if(!triangles[i].deleted)
			{
			/*	if (triangles[i].material != cur_material)
				{
					cur_material = triangles[i].material;
					fprintf(file, "usemtl %s\n", materials[triangles[i].material].c_str());
				}*/
				if (has_uv)
				{
					 //if(i<120 )qDebug()<<" ==>"<<triangles[i].v[0]+1<<" ,  "<< triangles[i].v[1]+1<<" , "<< triangles[i].v[2]+1;
					fprintf(file, "f %d/%d %d/%d %d/%d\n", triangles[i].v[0]+1, triangles[i].v[0]+1, triangles[i].v[1]+1, triangles[i].v[1]+1, triangles[i].v[2]+1, triangles[i].v[2]+1);
					//uv += 3;
				}
				else
				{
				   // if(i<120 )qDebug()<< triangles[i].v[0]+1<<" ,  "<< triangles[i].v[1]+1<<" , "<< triangles[i].v[2]+1;
					fprintf(file, "f %d %d %d\n", triangles[i].v[0]+1, triangles[i].v[1]+1, triangles[i].v[2]+1);
				}
				//fprintf(file, "f %d// %d// %d//\n", triangles[i].v[0]+1, triangles[i].v[1]+1, triangles[i].v[2]+1); //more compact: remove trailing zeros
			}
		}
		fclose(file);

	}

int SurfaceMesh::getNumVertices()
{
    return m_positionAttribute->count();
}

Qt3DCore::QBuffer* SurfaceMesh::getVertexBuffer()
{

    return m_vertexBuffer;
}

Qt3DCore::QBuffer* SurfaceMesh::getNormalBuffer()
{
    return m_normalBuffer;
}

int SurfaceMesh::getWidth()
{
    return m_width;
}
int SurfaceMesh::getHeight()
{
    return m_height;
}

void SurfaceMesh::activateNullValue(float nullValue)
{
	m_nullValueActive = true;
	m_nullValue = nullValue;
}

void SurfaceMesh::deactivateNullValue()
{
	m_nullValueActive = false;
}

float SurfaceMesh::nullValue() const
{
	return m_nullValue;
}

bool SurfaceMesh::isNullValueActive() const
{
	return m_nullValueActive;
}

void SurfaceMesh::filterTrianglePoints(int i, const QVector3D& p1, const QVector3D& p2,
		const QVector3D& p3, const uint* rawIndexArray, const float* rawVertexArray, QVector<MinMax>& listeindex,
		QVector3D& n1) {
	bool p1Null = p1.y()==m_internalNullValue;
	bool p2Null = p2.y()==m_internalNullValue;
	bool p3Null = p3.y()==m_internalNullValue;

	QVector3D errorNormal = QVector3D(0.0f,-1.0f,0.0f);
	if (!p1Null && !p2Null && !p3Null)
	{
		float angle = 180.0f / 3.14159f * acos(QVector3D::dotProduct(n1,QVector3D(0,-1,0)));
		if (angle > 87.0f)
		{
			int min=-1;
			int max=-1;
			int other =-1;

			if( p1.y()>p2.y())
			{
				if(p2.y()<p3.y())
				{
					min = rawIndexArray[i+1]*3+1;
					other = rawIndexArray[i]*3+1;
				}
				else
				{
					min = rawIndexArray[i+2]*3+1;
					other = rawIndexArray[i]*3+1;
				}
				if(p1.y()>p3.y())
				{
					//P1 est trop loin
					max = rawIndexArray[i]*3  +1;
					//qDebug()<<" max p1:"<<p1;
				}
				else
				{
					//P3 est trop loin
					max = rawIndexArray[i+2]*3+1;
					//qDebug()<<" max p3:"<<p3;
				}
			}
			else
			{
				if(p1.y()<p3.y())
				{
					min = rawIndexArray[i]*3  +1;
					other = rawIndexArray[i+1]*3+1;
				}
				else
				{
					min = rawIndexArray[i+2]*3+1;
					other = rawIndexArray[i+1]*3+1;
				}
				if(p2.y()>p3.y())
				{
					//P2 est trop loin
					max = rawIndexArray[i+1]*3+1;
					//qDebug()<<" max p2:"<<p2;
				}
				else
				{
					//P3 est trop loin
					max = rawIndexArray[i+2]*3+1;
				//	qDebug()<<" max p3:"<<p3;
				}
			}

			float dist = rawVertexArray[max] -rawVertexArray[min];
			float dist2 = rawVertexArray[other] -rawVertexArray[min];
			//qDebug()<<" distance max : "<<dist<<" distance milieu : "<<dist2;
			if( dist > 600 )
			{
				n1 = errorNormal;
				if (dist2>1) listeindex.push_back(MinMax(other, rawVertexArray[min]));
				listeindex.push_back(MinMax(max, rawVertexArray[min]));
			}
		}
	}
	else if (!p1Null && !p2Null && p3Null)
	{
		int index1 = rawIndexArray[i]*3  +1;
		int index2 = rawIndexArray[i+1]*3+1;
		int index3 = rawIndexArray[i+2]*3+1;
		int min, max;
		float dist = std::fabs(rawVertexArray[index1] - rawVertexArray[index2]);
		if (rawVertexArray[index1]<rawVertexArray[index2])
		{
			min = index1;
			max = index2;
		}
		else
		{
			min = index2;
			max = index1;
		}

		if (dist>600)
		{
			listeindex.push_back(MinMax(max, rawVertexArray[min]));
		}
		listeindex.push_back(MinMax(index3, rawVertexArray[min]));
		n1 = errorNormal;
	}
	else if (!p1Null && p2Null && !p3Null)
	{
		int index1 = rawIndexArray[i]*3  +1;
		int index2 = rawIndexArray[i+1]*3+1;
		int index3 = rawIndexArray[i+2]*3+1;
		int min, max;
		float dist = std::fabs(rawVertexArray[index1] - rawVertexArray[index3]);
		if (rawVertexArray[index1]<rawVertexArray[index3])
		{
			min = index1;
			max = index3;
		}
		else
		{
			min = index3;
			max = index1;
		}

		if (dist>600)
		{
			listeindex.push_back(MinMax(max, rawVertexArray[min]));
		}
		listeindex.push_back(MinMax(index2, rawVertexArray[min]));
		n1 = errorNormal;
	}
	else if (p1Null && !p2Null && !p3Null)
	{
		int index1 = rawIndexArray[i]*3  +1;
		int index2 = rawIndexArray[i+1]*3+1;
		int index3 = rawIndexArray[i+2]*3+1;
		int min, max;
		float dist = std::fabs(rawVertexArray[index3] - rawVertexArray[index2]);
		if (rawVertexArray[index3]<rawVertexArray[index2])
		{
			min = index3;
			max = index2;
		}
		else
		{
			min = index2;
			max = index3;
		}

		if (dist>600)
		{
			listeindex.push_back(MinMax(max, rawVertexArray[min]));
		}
		listeindex.push_back(MinMax(index1, rawVertexArray[min]));
		n1 = errorNormal;
	}
	else if (!p1Null)
	{
		int index1 = rawIndexArray[i]*3  +1;
		int index2 = rawIndexArray[i+1]*3+1;
		int index3 = rawIndexArray[i+2]*3+1;

		listeindex.push_back(MinMax(index2, rawVertexArray[index1]));
		listeindex.push_back(MinMax(index3, rawVertexArray[index1]));
		n1 = errorNormal;
	}
	else if (!p2Null)
	{
		int index1 = rawIndexArray[i]*3  +1;
		int index2 = rawIndexArray[i+1]*3+1;
		int index3 = rawIndexArray[i+2]*3+1;

		listeindex.push_back(MinMax(index1, rawVertexArray[index2]));
		listeindex.push_back(MinMax(index3, rawVertexArray[index2]));
		n1 = errorNormal;
	}
	else if (!p3Null)
	{
		int index1 = rawIndexArray[i]*3  +1;
		int index2 = rawIndexArray[i+1]*3+1;
		int index3 = rawIndexArray[i+2]*3+1;

		listeindex.push_back(MinMax(index2, rawVertexArray[index3]));
		listeindex.push_back(MinMax(index1, rawVertexArray[index3]));
		n1 = errorNormal;
	}
	else
	{
		n1 = errorNormal;
	}
}
