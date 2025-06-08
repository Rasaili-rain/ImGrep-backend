from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def health_check():
    """API health check"""
    return Response({'status': 'ok', 'service': 'image-search-backend'})

@api_view(['GET', 'POST'])
def test_api(request):
    """Test API endpoint"""
    if request.method == 'GET':
        return Response({'message': 'API is working', 'method': 'GET'})
    
    return Response({
        'message': 'Data received',
        'method': 'POST',
        'data': request.data
    })